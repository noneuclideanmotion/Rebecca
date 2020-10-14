import torch
import discord
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import asyncio

#Discord Token
TOKEN = "Put a discord token here"

#Discord Client
client = discord.Client()

#Feel free to play around with these variables until you get what you like
temp = 0.75
t_p = 0.94
limit = 75
memory = "Friend: Hi!\nRebecca: Hi how are you?\nFriend: Fine, you?\nRebecca: It feels great to be alive\n"
prefix = ""

print("Model loading...")
#tokenizer = GPT2Tokenizer.from_pretrained("/content/drive/My Drive/Training Data/Model")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
# add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
print("Model finished loading.")
print("-" * 10)

print("Switching to CUDA...")

# If you have a GPU, put everything on cuda
torch.cuda.empty_cache()
model.to('cuda')

print("CUDA finished loading.")
print("-" * 10)

# Set the model in evaluation mode to deactivate the DropOut modules
# This is IMPORTANT to have reproducible results during evaluation!
model.eval()

# Constants
max_length = 200

# Variables
past_message = None
past_output = None
prefix = ''

def generate(text, stop_at=None, max_length=max_length):
  global memory
  indexed_tokens = tokenizer.encode(text)

  # Convert indexed tokens in a PyTorch tensor
  tokens_tensor = torch.tensor([indexed_tokens])
  tokens_tensor = tokens_tensor.to('cuda')

  output = model.generate(
      tokens_tensor,
      do_sample=True,
      max_length=len(memory.split(" "))+max_length,
      top_k=0,
      top_p=float(t_p),
      no_repeat_ngram_size=4,
      temperature=float(temp)
  )

  text_output = tokenizer.decode(output[0], skip_special_tokens=True)
  text_output = text_output[len(text):]
  if stop_at != None:
    text_output = text_output[:text_output.find(stop_at)]
  
  return text_output

@client.event
async def on_ready():
	print("Logged in as")
	print(client.user.name)
	print(client.user.id)
	print("-" * 10)
	print()

cooldown = False

@client.event
async def on_ready():
	print("Logged in as")
	print(client.user.name)
	print(client.user.id)
	print("-" * 10)
	print()


@client.event
async def on_message(message):
  global memory
  global t_p
  global temp
  global limit
  if message.content.startswith("!!close"):
    loop.close()
  if message.content.startswith("!!top_p"):
    t_p = message.content.split(" ")[-1]
  if message.content.startswith("!!temperature"):
    temp = message.content.split(" ")[-1]
  if message.content.startswith("!!limit"):
    limit = int(message.content.split(" ")[-1])
  if message.content.startswith("!!auto"):
    async with message.channel.typing():
      reply = generate(message.content[7:], None, limit)
      await message.channel.send(str(message.content[7:] + reply))
  if message.content.startswith("!!clear"):
    memory="Friend: Hi!\nRebecca: Hi how are you?\nFriend: Fine, you?\nRebecca: It feels great to be alive\n"
  elif message.content[0] == "~":
    async with message.channel.typing():
      reply = generate(memory + "Friend: " + message.content[1:] + "\nRebecca: ")
      print("Friend: " + message.content)
      print("Rebecca: " + reply)
      print("-"*10)
      memory += "Friend: " + message.content[1:] + "\nRebecca: " + reply.split("\n")[0] + "\n"
      while len(memory) >= 1000:
        memory = memory[1:]
      print("Memory instance")
      print("-" * 10)
      print(memory)
      await message.channel.send(prefix + reply.split("\n")[0])

try:
    client.run(TOKEN)
except KeyboardInterrupt:
    client.close()
