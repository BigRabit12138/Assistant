import sys

from pathlib import Path
from torch import float16
from loguru import logger


##################################################################
# 目录设置
# project dictionary
project_dir = Path.cwd()

# workspace root
workspace_root = project_dir / 'workspace'

# prompt path
prompt_path = project_dir / 'agents/prompts'

# tmp
tmp = project_dir / 'tmp'

# data path
data_path = project_dir / 'data'

# 目录设置结束
###############################################################


###########################################################
# 日志对象设置
# logger
logger_path = project_dir / 'logs/log.txt'
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(logger_path, level="DEBUG")
# 日志对象设置结束
################################################################


###############################################################
# 代理设置
# global proxy
global_proxy = ''

# 全局代理设置的参数
proxy_setting = {
    'addr': '127.0.0.1',
    'port': 10808
}

# 默认无代理的socket, 在main.py中设置
default_socket = None
# 代理设置结束
################################################################


#####################################################
# claude api 设置
# if Anthropic
# claude api key official
claude_api_key = ''

# claude2 非官方api接口，使用Web的sessionKey发送请求、接受响应
# 详情访问https://github.com/AshwinPathi/claude-api-py
claude_api_sessionKey =\
    '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'
# claude api 设置结束
########################################################


########################################################################
# 记忆设置
# for Execution
# long term memory
long_term_memory = True

# memory ttl
mem_ttl = 24 * 30 * 3600

# chroma database
embedding_function = 'moka-ai/m3e-base'

# splitter chunk size
chunk_size = 512
# 记忆设置结束
#########################################################################


#########################################################################
# huggingface 的chat api设置
# huggingchat
email = 'wuzhenglingame@gmail.com'
passwd = 'qazedctgb13579QAZ'
# huggingface 的chat api设置结束
#######################################################################


########################################################################
# 本地模型运行设置
# device
device = 'cuda'

# torch type: torch.float16
torch_type = float16
# 本地模型运行设置结束
##########################################################################


##########################################################################
# 是否在本地运行，在本地测试的时候为True，会在连接本地Websockets服务器是屏蔽代理
run_local_mode = False

if run_local_mode:
    ip = '127.0.0.1'
    port = 9999
else:
    ip = '45.76.66.110'
    port = 9999
############################################################################
