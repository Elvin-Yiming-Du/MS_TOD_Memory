import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Each query needs to be accompanied by an corresponding instruction describing the task.
task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",}

query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
queries = [
    'are judo throws allowed in wrestling?', 
    'how to become a radiology technician in michigan?'
    ]

# No instruction needed for retrieval passages
passage_prefix = ""
passages = [
    "Since you're reading this, you are probably someone from a judo background or someone who is just wondering how judo techniques can be applied under wrestling rules. So without further ado, let's get to the question. Are Judo throws allowed in wrestling? Yes, judo throws are allowed in freestyle and folkstyle wrestling. You only need to be careful to follow the slam rules when executing judo throws. In wrestling, a slam is lifting and returning an opponent to the mat with unnecessary force.",
    "Below are the basic steps to becoming a radiologic technologist in Michigan:Earn a high school diploma. As with most careers in health care, a high school education is the first step to finding entry-level employment. Taking classes in math and science, such as anatomy, biology, chemistry, physiology, and physics, can help prepare students for their college studies and future careers.Earn an associate degree. Entry-level radiologic positions typically require at least an Associate of Applied Science. Before enrolling in one of these degree programs, students should make sure it has been properly accredited by the Joint Review Committee on Education in Radiologic Technology (JRCERT).Get licensed or certified in the state of Michigan."
]

# load model with tokenizer
model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True)

# get the embeddings
max_length = 32768
query_embeddings = model.encode(queries, instruction=query_prefix, max_length=max_length)
passage_embeddings = model.encode(passages, instruction=passage_prefix, max_length=max_length)

# normalize embeddings
query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)

# get the embeddings with DataLoader (spliting the datasets into multiple mini-batches)
# batch_size=2
# query_embeddings = model._do_encode(queries, batch_size=batch_size, instruction=query_prefix, max_length=max_length, num_workers=32, return_numpy=True)
# passage_embeddings = model._do_encode(passages, batch_size=batch_size, instruction=passage_prefix, max_length=max_length, num_workers=32, return_numpy=True)

scores = (query_embeddings @ passage_embeddings.T) * 100
print(scores.tolist())
# [[87.42693328857422, 0.46283677220344543], [0.965264618396759, 86.03721618652344]]
