import matplotlib.pyplot as plt

# #折线图
x = [1.0, 2.0, 3.0, 4.0]#点的横坐标
k1 = [0.5,0.1,0.1,0.05]#线1的纵坐标
k2 = [1.0,0.5,0.1,0.05]#线1的纵坐标
x3 = [0.5, 1.0, 1.5, 2.0]#点的横坐标
k3 = [1.0,1.0,1.0,0.5]#线1的纵坐标
# values = [0.5,0.1,0.1,0.05]
#8ECFC9
#FFBE7A
#FA7F6F
# k2 = [0.8988,0.9334,0.9435,0.9407,0.9453,0.9453]#线2的纵坐标
plt.figure()
plt.title("The optimal combination result of margin and scale")#图例
plt.plot(x,k1,'s-',markersize=10,color = '#8ECFC9',label="HarM")#s-:方形
plt.plot(x,k2,'o-',markersize=8, color = '#FFBE7A',label="MAMI")#s-:方形
plt.plot(x3,k3,'*-',color = '#FA7F6F',label="FHM")#s-:方形
# plt.plot(x,k2,'o-',color = 'g',label="CNN-RLSTM")#o-:圆形
plt.xlabel("margin " + chr(947))#横坐标名字
plt.ylabel("scale "+chr(945))#纵坐标名字
# plt.xticks(values,k1)
plt.ylim(0,1.2)
# for a, b in zip(x, k1):  
#     plt.text(a, b, (a, b))

# for a, b in zip(x, k2):  
#     plt.text(a, b, (a, b))

# for a, b in zip(x3, k3):  
#     plt.text(a, b, (a, b))
plt.legend()

plt.savefig("z-harm-mami-fhm.png")

#折线图
# x = [1.0, 2.0, 3.0, 4.0]#点的横坐标
# k1 = [1.0,0.5,0.1,0.05]#线1的纵坐标
# values = [0.5,0.1,0.1,0.05]

# k2 = [0.8988,0.9334,0.9435,0.9407,0.9453,0.9453]#线2的纵坐标
# plt.figure()
# plt.title("The optimal combination result of margin and scale. (MAMI)")#图例
# plt.plot(x,k1,'s-',color = '#FFBE7A',label="MAMI")#s-:方形
# # plt.plot(x,k2,'o-',color = 'g',label="CNN-RLSTM")#o-:圆形
# plt.xlabel("margin " + chr(947))#横坐标名字
# plt.ylabel("scale "+chr(945))#纵坐标名字
# # plt.xticks(values,k1)
# for a, b in zip(x, k1):  
#     plt.text(a, b, (a, b))
# plt.legend()

# plt.savefig("z-mami.png")

# x = [0.5, 1.0, 1.5, 2.0]#点的横坐标
# k1 = [1.0,1.0,1.0,0.5]#线1的纵坐标
# values = [0.5,0.1,0.1,0.05]

# k2 = [0.8988,0.9334,0.9435,0.9407,0.9453,0.9453]#线2的纵坐标
# plt.figure()
# plt.title("The optimal combination result of margin and scale")#图例
# plt.plot(x,k1,'s-',color = '#FA7F6F',label="FHM")#s-:方形
# # plt.plot(x,k2,'o-',color = 'g',label="CNN-RLSTM")#o-:圆形
# plt.xlabel("margin " + chr(947))#横坐标名字
# plt.ylabel("scale "+chr(945))#纵坐标名字
# # plt.xticks(values,k1)
# for a, b in zip(x, k1):  
#     plt.text(a, b, (a, b))
# plt.legend()

# plt.ylim(0,2)

# plt.savefig("z-fhm.png")