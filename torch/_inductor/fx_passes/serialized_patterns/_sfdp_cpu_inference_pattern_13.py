# noqa: F401, E501
# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python
# torchgen/fuse_attention_patterns/gen_attention_patterns.py

import torch
import torch._inductor

aten = torch.ops.aten
prims = torch.ops.prims

from torch._inductor.pattern_matcher import (
   Arg,
   CallFunction,
   CallFunctionVarArgs,
   CallMethod,
   CallMethodVarArgs,
   CallModule,
   CallModuleVarArgs,
   ExclusiveKeywordArg,
   Ignored,
   KeywordArg,
   ListOf,
   MultiOutputPattern,
   PatternExpr,
   RepeatedExpr,
   _TargetArgsExpr,
   _TargetExpr,
   _TargetExprVarArgs,
)
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
bmm_default = CallFunction(aten.bmm.default, KeywordArg('query'), permute_default, _users=2)
amax_default = CallFunction(aten.amax.default, bmm_default, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, bmm_default, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
clone_default = CallFunction(aten.clone.default, div_Tensor)
_sfdp_cpu_inference_pattern_13 = CallFunction(aten.bmm.default, clone_default, KeywordArg('value'))


permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
bmm_default = CallFunction(aten.bmm.default, KeywordArg('query'), permute_default)
convert_element_type_default = CallFunction(prims.convert_element_type.default, bmm_default, Ignored(), _users=2)
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored())
clone_default = CallFunction(aten.clone.default, convert_element_type_default_1)
_sfdp_cpu_inference_pattern_13_bfloat16 = CallFunction(aten.bmm.default, clone_default, KeywordArg('value'))
