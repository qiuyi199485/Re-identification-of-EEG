import pandas as pd
import matplotlib.pyplot as plt

# 创建示例数据
data = {
    'Category A': [30, 43, 30, 40, 40],
    'Category B': [15, 25, 35, 45, 55],
    'Category C': [20, 30, 40, 50, 60]
}

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 绘制箱线图
plt.figure(figsize=(8, 6))

# 定义箱线图的颜色
boxprops = dict(color="black", linewidth=1.5)
medianprops = dict(color="black", linewidth=2)
whiskerprops = dict(color="black", linewidth=1.5)
capprops = dict(color="black", linewidth=1.5)
flierprops = dict(markeredgecolor="black")

# 绘制每个类别的箱线图，并设置颜色
bplot = plt.boxplot(
    [df['Category A'], df['Category B'], df['Category C']],
    patch_artist=True,
    boxprops=boxprops,
    medianprops=medianprops,
    whiskerprops=whiskerprops,
    capprops=capprops,
    flierprops=flierprops
)

# 设置颜色
colors = ['#EC6602', '#009999', 'gray']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

# 添加图例
plt.legend([bplot["boxes"][0], bplot["boxes"][1], bplot["boxes"][2]],
           ['Category A', 'Category B', 'Category C'],
           loc='upper right')


# 添加标题和标签
#plt.title('Boxplot of Categories')
#plt.xlabel('Category')
plt.ylabel('Accuracy')

# 显示图形
plt.show()
