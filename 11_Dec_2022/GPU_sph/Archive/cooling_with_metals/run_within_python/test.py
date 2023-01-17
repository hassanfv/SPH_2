import subprocess 

subprocess.call(["g++", "test.cpp"]) 
tmp=subprocess.call("./a.out")

print(tmp)
