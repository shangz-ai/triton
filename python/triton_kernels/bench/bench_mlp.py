from itertools import chain
from pathlib import Path
from copy import deepcopy
import triton.profiler as proton
import torch
import argparse
import triton_kernels
import triton_kernels.roofline as roofline
import triton_kernels.swiglu
from triton_kernels.matmul_ogs import matmul_ogs, PrecisionConfig, FlexCtx, FnSpecs, FusedActivation
from triton_kernels.matmul_ogs_details.opt_flags import make_opt_flags, reset_opt_flags_constraints, update_opt_flags_constraints, scoped_opt_flags_constraints
from triton_kernels.tensor import Storage, convert_layout, Tensor
from triton_kernels.target_info import get_cdna_version
import distributed as triton_dist
from triton_kernels.tensor_details import layout
from triton_kernels.tensor_details.layout import BlackwellMX4ValueShuffledLayout
from bench_utils import quantize_weight
import tempfile


def _infer_opt_flags(x, w, rdata, pc, gather_indx=None, scatter_indx=None):
    """
    Infer opt_flags by calling make_opt_flags with the same parameters matmul_ogs would use.
    This ensures the block shapes match what the kernel will actually select.
    """
    if not isinstance(w, Tensor):
        raise TypeError("w must be a Tensor for block shape inference")

    # Get dimensions - weight is [E, K, N] where K is the reduction dim
    K = w.shape[-2]
    N = w.shape[-1]
    M = x.shape[-2] if gather_indx is None else gather_indx.src_indx.shape[0]
    batch_size = w.shape[0] if (rdata is None or rdata.expt_hist is None) and w.ndim == 3 else 1
    has_scatter = scatter_indx is not None

    out_dtype = pc.out_dtype or x.dtype
    x_transpose = x.stride(-1) != 1

    x_storage = Storage(x)
    w_storage = w.storage
    w_scale = pc.weight_scale
    if w_scale is None:
        w_scale_ok = True
    elif isinstance(w_scale, Tensor):
        w_scale_ok = w_scale.storage.is_tma_compliant()
    else:
        w_scale_ok = Storage(w_scale).is_tma_compliant()

    can_use_tma = x_storage.is_tma_compliant() and w_storage.is_tma_compliant() and w_scale_ok
    can_use_fused_scatter = has_scatter and rdata is not None and rdata.n_expts_act == 1

    # Respects any constraints set by the caller via update_opt_flags_constraints
    opt_flags = make_opt_flags(
        out_dtype=out_dtype,
        lhs_dtype=x.dtype,
        rhs_dtype=w.dtype,
        precision_config=pc,
        batch_size=batch_size,
        m=M,
        n=N,
        k=K,
        routing_data=rdata,
        can_use_persistent_tma=can_use_tma,
        can_use_fused_scatter=can_use_fused_scatter,
        epilogue_effective_itemsize=None,
        x_transpose=x_transpose,
        has_y_acc_in=False,
        block_k=None,
    )
    return opt_flags


def bench_mlp(batch_per_expt, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP,
              shuffle_mx4=False, num_stages_fc1=None, num_stages_fc2=None,
              epilogue_subtile_fc1=None):
    assert n_expts_tot % EP == 0
    assert dim2 % TP == 0
    rank, world_size = triton_dist.setup()
    dev = f"cuda:{rank}"
    DP = world_size
    batch = batch_per_expt * n_expts_tot // n_expts_act

    assert n_expts_tot % EP == 0, f"{n_expts_tot=}, {EP=}, n_expts_tot must be divisible by EP"
    assert dim2 % TP == 0, f"{dim2=}, {TP=}, dim2 must be divisible by TP"

    # -- init data --
    # weights
    wg = triton_dist.broadcast(torch.randn((dim1, n_expts_tot), device=dev))
    w1 = torch.randn((n_expts_tot // EP, dim1, dim2 // TP), device=dev)
    w2 = torch.randn((n_expts_tot // EP, dim2 // TP // 2, dim1), device=dev)
    # biases
    bg = triton_dist.broadcast(torch.randn((n_expts_tot, ), device=dev))
    b1 = torch.randn((n_expts_tot // EP, dim2 // TP), device=dev)
    b2 = torch.randn((n_expts_tot // EP, dim1), device=dev)
    ep_indx = (rank // TP) % EP
    groups = [list(range(ep * TP, (ep + 1) * TP)) for ep in range(EP)]
    b2 = triton_dist.broadcast(b2, src=ep_indx * TP, groups=groups, group_idx=ep_indx)

    # -- numerics --
    opt1 = dict()
    opt2 = dict()
    if w_dtype == "mx4":
        num_warps = 4 if batch <= 512 else 8
        value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(mx_axis=1)
        scale_layout, scale_layout_opts = layout.make_default_matmul_mxfp4_w_scale_layout(
            mx_axis=1, num_warps=num_warps)
        opt1 = {
            "value_layout": value_layout,
            "value_layout_opts": value_layout_opts,
            "scale_layout": scale_layout,
            "scale_layout_opts": scale_layout_opts,
        }
        opt2 = deepcopy(opt1)
    wg, wg_flex, wg_scale = quantize_weight(wg, "bf16")
    w1, w1_flex, w1_scale = quantize_weight(w1, w_dtype, **opt1)
    w2, w2_flex, w2_scale = quantize_weight(w2, w_dtype, **opt2)
    pcg = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=wg_flex), weight_scale=wg_scale)
    act = FusedActivation(FnSpecs("swiglu", triton_kernels.swiglu.swiglu_fn, ("alpha", "limit")), (1.0, 1.0), 2)
    pc1 = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=w1_flex), weight_scale=w1_scale)
    pc2 = PrecisionConfig(flex_ctx=FlexCtx(rhs_data=w2_flex), weight_scale=w2_scale)

    # -- benchmark --
    x_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}[x_dtype]
    # special treatment of fp8_e4m3 on AMD CDNA3 because it uses fp8_e4m3fnuz
    if x_dtype == torch.float8_e4m3fn and get_cdna_version() == 3:
        x_dtype = torch.float8_e4m3fnuz

    input_x = torch.randn((batch // DP, dim1), device=dev)
    input_x = input_x.to(x_dtype)
    xg = input_x.to(wg.dtype if n_expts_tot > 1 else input_x.dtype)

    # For MX4 shuffling: run one dry-run iteration to collect routing data, then infer block shapes
    if shuffle_mx4 and w_dtype == "mx4":
        # Block swap is a optimization for mx4 weight layout for better cacheline behavior; shuffled
        # weights have contiguous tiles, making the swap unnecessary.
        dry_run_constraints = {"disable_mx4_block_swap": True}
        if epilogue_subtile_fc1 is not None:
            dry_run_constraints["epilogue_subtile"] = epilogue_subtile_fc1
        with scoped_opt_flags_constraints(dry_run_constraints):
            # Dry-run to get routing data
            if n_expts_tot > 1:
                logits = matmul_ogs(xg, wg, bg, precision_config=pcg)
                x_dry, rdata_dry, gather_indx_dry, scatter_indx_dry, _ = triton_dist.routing(
                    input_x, logits, n_expts_act, EP=EP, TP=TP)
            else:
                x_dry = triton_dist.all_gather(input_x, dim=0)
                rdata_dry, gather_indx_dry, scatter_indx_dry = None, None, None

            if x_dry.nelement() > 0:
                # Infer block shapes for W1 (has gather_indx)
                opt_flags_w1 = _infer_opt_flags(x_dry, w1, rdata_dry, pc1, gather_indx=gather_indx_dry)
                w1_block_k, w1_block_n = opt_flags_w1.block_k, opt_flags_w1.block_n
                w1 = convert_layout(w1, BlackwellMX4ValueShuffledLayout, block_k=w1_block_k, block_n=w1_block_n)

                # Run W1 once to get y_dry for W2 block shape inference
                y_dry = matmul_ogs(x_dry, w1, b1, rdata_dry, gather_indx=gather_indx_dry,
                                   precision_config=pc1, fused_activation=act)

                # Infer block shapes for W2 (has scatter_indx)
                opt_flags_w2 = _infer_opt_flags(y_dry, w2, rdata_dry, pc2, scatter_indx=scatter_indx_dry)
                w2_block_k, w2_block_n = opt_flags_w2.block_k, opt_flags_w2.block_n
                w2 = convert_layout(w2, BlackwellMX4ValueShuffledLayout, block_k=w2_block_k, block_n=w2_block_n)

                print(f"Shuffled layout: FC1 block_k={w1_block_k}, block_n={w1_block_n}; "
                      f"FC2 block_k={w2_block_k}, block_n={w2_block_n}")
        torch.cuda.synchronize()

    # run layer
    fpath = Path(tempfile.mktemp())
    proton.start(str(fpath), hook="triton")
    for i in range(100):
        if n_expts_tot > 1:  # sparse
            logits = matmul_ogs(xg, wg, bg, precision_config=pcg)
            x, rdata, gather_indx, scatter_indx, metadata = triton_dist.routing(input_x, logits, n_expts_act, EP=EP,
                                                                                TP=TP)
        else:  # dense
            x = triton_dist.all_gather(input_x, dim=0)
            rdata, gather_indx, scatter_indx, metadata = None, None, None, None
        if x.nelement() > 0:
            fc1_constraints = {}
            if shuffle_mx4:
                fc1_constraints["disable_mx4_block_swap"] = True
            if num_stages_fc1 is not None:
                fc1_constraints["num_stages"] = num_stages_fc1
            if epilogue_subtile_fc1 is not None:
                fc1_constraints["epilogue_subtile"] = epilogue_subtile_fc1
            with scoped_opt_flags_constraints(fc1_constraints):
                x = matmul_ogs(x, w1, b1, rdata, gather_indx=gather_indx, precision_config=pc1, fused_activation=act)
            fc2_constraints = {}
            if shuffle_mx4:
                fc2_constraints["disable_mx4_block_swap"] = True
            if num_stages_fc2 is not None:
                fc2_constraints["num_stages"] = num_stages_fc2
            with scoped_opt_flags_constraints(fc2_constraints):
                x = matmul_ogs(x, w2, b2 if rank % TP == 0 else None, rdata, scatter_indx=scatter_indx,
                               precision_config=pc2)
        x = triton_dist.reduce_scatter(x, metadata=metadata, dim=0)
    proton.finalize()

    return roofline.parse_profile(fpath.with_suffix(".hatchet"), useful_op_regex=".*matmul.*")


def roofline_mlp(batch_sizes, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP, \
                  name="", verbose=True, shuffle_mx4=False, num_stages_fc1=None,
                  num_stages_fc2=None, epilogue_subtile_fc1=None):
    suffix = "-shuffled" if shuffle_mx4 else ""
    suffix += f"-fc1stages{num_stages_fc1}" if num_stages_fc1 is not None else ""
    suffix += f"-fc2stages{num_stages_fc2}" if num_stages_fc2 is not None else ""
    suffix += f"-fc1subtile{epilogue_subtile_fc1}" if epilogue_subtile_fc1 is not None else ""
    out_path = Path(f"logs/{name}/{x_dtype}x-{w_dtype}w-TP{TP}-EP{EP}{suffix}/")
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = roofline.compute_roofline(dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, EP,  # fixed args
                                         shuffle_mx4=shuffle_mx4,  # weight shuffling option
                                         num_stages_fc1=num_stages_fc1,  # override num_stages for FC1 kernel
                                         num_stages_fc2=num_stages_fc2,  # override num_stages for FC2 kernel
                                         epilogue_subtile_fc1=epilogue_subtile_fc1,  # override epilogue subtile for FC1
                                         bench_fn=bench_mlp,  # function to benchmark
                                         intensity_proxy_name="batch_per_expt",  # intensity proxy name
                                         intensity_proxy_values=batch_sizes,  # intensity proxy values to sweep
                                         verbose=verbose,  # options
                                         out_path=out_path.with_suffix(".csv"))  # output path
    png_path = roofline.plot_roofline(series=[csv_path],  # roofline data to plot
                                      flops_dtype=x_dtype,  # dtype to use for FLOPS roof
                                      xlabel="batch_per_expt", title=out_path,  # plot option
                                      out_path=out_path.with_suffix(".png"),  # output path
                                      max_tbps="memset", max_tflops="cublas")  # hardware limits
    return png_path


if __name__ == "__main__":
    has_native_mx4 = torch.cuda.get_device_capability(0)[0] >= 10 or get_cdna_version() == 4
    batch_sizes_dense = [*range(128, 8192, 128)]
    batch_ranges_moe = [(2**(2 + k), 2**(3 + k), min(2**k, 32)) for k in range(8)]
    batch_sizes_moe = list(chain(*[range(*r) for r in batch_ranges_moe]))
    batch_sizes_moe = [1, 2, 4, 8, 16, 32, 64]
    dense_dtypes = ["fp8", "fp8"]
    quantized_dtypes = ["fp8", "mx4"] if has_native_mx4 else ["bf16", "mx4"]
    rank, world_size = triton_dist.setup()
    if world_size > 1:
        # Running all workloads at once may cause OOM on some GPUs such as H100 80GB.
        # Thus we request users to run each workload separately.
        # For example, all eligible combinations of options are listed below when four GPUs are used:
        # torchrun --nproc-per-node=4 ./bench_mlp.py --tp 2 --ep 2 --name gpt-oss-x2
        # torchrun --nproc-per-node=4 ./bench_mlp.py --tp 1 --ep 4 --name gpt-oss-x2
        # torchrun --nproc-per-node=4 ./bench_mlp.py --tp 4 --ep 1 --name gpt-oss-x2
        # torchrun --nproc-per-node=4 ./bench_mlp.py --tp 4 --ep 1 --name dense
        # torchrun --nproc-per-node=4 ./bench_mlp.py --tp 2 --ep 2 --name gpt-oss-x2 --quantized
        # torchrun --nproc-per-node=4 ./bench_mlp.py --tp 1 --ep 4 --name gpt-oss-x2 --quantized
        # torchrun --nproc-per-node=4 ./bench_mlp.py --tp 4 --ep 1 --name gpt-oss-x2 --quantized
        # torchrun --nproc-per-node=4 ./bench_mlp.py --tp 4 --ep 1 --name dense --quantized
        parser = argparse.ArgumentParser()
        parser.add_argument("--tp", type=int, default=1)
        parser.add_argument("--ep", type=int, default=1)
        parser.add_argument("--name", type=str, choices=["dense", "gpt-oss-x2"])
        parser.add_argument("--quantized", action="store_true", default=False)
        args = parser.parse_args()
        dtypes = quantized_dtypes if args.quantized else dense_dtypes
        if args.name == "dense":
            assert args.ep == 1, "EP must be 1 for dense"
            roofline_mlp(batch_sizes_dense, 8192, 8192, 1, 1, dtypes[0], dtypes[1], TP=args.tp, EP=args.ep,
                         name="dense")
        else:
            roofline_mlp(batch_sizes_moe, 5760, 5760, 128, 4, dtypes[0], dtypes[1], TP=args.tp, EP=args.ep,
                         name="gpt-oss-x2")
        triton_dist.cleanup()
    else:
        # Common args for all MoE scenarios
        moe_args = dict(dim1=5760, dim2=5760, n_expts_tot=128, n_expts_act=4, TP=1, EP=1, name="gpt-oss-x2")

        # Run 5 scenarios to isolate 3 optimization categories:
        #   Shuffling: scenario 2 → 3
        #   FC2 5stg:  scenario 3 → 4
        #   FC1 5stg:  scenario 4 → 5

        # 1. FP8 baseline
        roofline_mlp(batch_sizes_moe, x_dtype=dense_dtypes[0], w_dtype=dense_dtypes[1], **moe_args)
        # 2. MX4 baseline
        roofline_mlp(batch_sizes_moe, x_dtype=quantized_dtypes[0], w_dtype=quantized_dtypes[1], **moe_args)
        # 3. MX4 shuffled
        roofline_mlp(batch_sizes_moe, x_dtype=quantized_dtypes[0], w_dtype=quantized_dtypes[1],
                     shuffle_mx4=True, **moe_args)
        # 4. MX4 shuffled + FC2 5stg
        roofline_mlp(batch_sizes_moe, x_dtype=quantized_dtypes[0], w_dtype=quantized_dtypes[1],
                     shuffle_mx4=True, num_stages_fc2=5, **moe_args)
        # 5. MX4 shuffled + subtile2 + FC1&FC2 5stg
        roofline_mlp(batch_sizes_moe, x_dtype=quantized_dtypes[0], w_dtype=quantized_dtypes[1],
                     shuffle_mx4=True, epilogue_subtile_fc1=2,
                     num_stages_fc1=5, num_stages_fc2=5, **moe_args)
