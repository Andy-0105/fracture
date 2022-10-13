import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
data = [json.loads(line) for line in open('metrics.json','r', encoding='utf-8')]
loss_data=[]
print(len(data))
for i in range(len(data)-1):
    losses=data[i]["total_loss"]
    loss_data.append(losses)
min_loss=min(loss_data)
max_loss=max(loss_data)
print(min_loss,max_loss)
avg_loss=(min_loss+max_loss)/4
x = range(len(loss_data))
y = loss_data
plt.xlabel("Steps")
plt.ylabel("loss")
plt.ylim(min_loss,avg_loss)
plt.plot(x,y, color = "coral")
x_locator=MultipleLocator(5000)
y_locator=MultipleLocator(0.05)
ax=plt.gca()
ax.xaxis.set_major_locator(x_locator)
ax.yaxis.set_major_locator(y_locator)
plt.legend()
plt.show()
