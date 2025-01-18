
import pandas as pd

a = pd.DataFrame([['1', 'a'], ['2', 'b']], columns=['num', 'alpha'])
b = a.groupby(lambda _x: True)
for i, row in a.iterrows():
    print(i)
    print(row)
    b = row['num']
    pass
import networkx as nx
import matplotlib.pyplot as plt

# 创建一个无向图
G = nx.Graph()

# 添加节点和边
G.add_node("A")
G.add_nodes_from(["B", "C", "D"])
G.add_edge("A", "B")
G.add_edges_from([("A", "C"), ("B", "D")])

# 打印节点和边
print("节点:", G.nodes())
print("边:", G.edges())

# 绘制图
nx.draw(G, with_labels=True)
plt.show()
pass
