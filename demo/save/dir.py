import os
rate=4
for i in range(1,11):
    if i==10:
        os.makedirs(f"./new48time/{rate}/{i}")
    else:
        os.makedirs(f"./new48time/{rate}/0{i}")

