import numpy as np
from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_runtime


loaded_json = open(temp.relpath("deploy_graph.json")).read()
loaded_lib = tvm.runtime.load_module(path_lib)
loaded_params = bytearray(open(temp.relpath("deploy_param.params"), "rb").read())
input_data = tvm.nd.array(np.random.uniform(size=data_shape).astype("float32"))

module = graph_runtime.create(loaded_json, loaded_lib, ctx)
module.load_params(loaded_params)
module.run(data=input_data)
out_deploy = module.get_output(0).asnumpy()
