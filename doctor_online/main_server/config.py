# 设置redis相关的配置信息
REDIS_CONFIG = {
	"host": "0.0.0.0",
	"port": 6379
}

# 设置neo4j图数据库的配置信息
NEO4J_CONFIG = {
	"uri": "bolt://0.0.0.0:7687",
	"auth": ("neo4j", "Itcast2019"),
	"encrypted": False
}

# 设置句子相关服务的请求地址
model_serve_url = "http://0.0.0.0:5001/v1/recognition/"

# 设置服务的超时时间
TIMEOUT = 2

# 设置规则对话的模板加载路径
reply_path = "./reply.json"

# 用户对话信息保存的过期时间
ex_time = 36000

