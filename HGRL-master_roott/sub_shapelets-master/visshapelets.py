
for i in range(shapelets.shape[0]):
    for j in range(shapelets.shape[1]):
        # 提取第三维度的第一个 shapelet
        first_shapelet = shapelets[i,j,  :]  # 选择第一个 shapelet

        # 可视化
        plt.figure(figsize=(10, 6))
        plt.plot(first_shapelet)
        plt.title('Visualization of the First Shapelet')
        plt.xlabel('Time Points')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.show()


shapelet_transform = learning_shapelets.transform(X_test)
dist_s1 = shapelet_transform[:, 0]
dist_s2 = shapelet_transform[:, 1]
print('dist_s1, dist_s2',dist_s1.shape, dist_s2.shape)
weights, biases = learning_shapelets.get_weights_linear_layer()

#(3, 2, 64)
fig = pyplot.figure(facecolor='white')
fig.set_size_inches(20, 8)
gs = fig.add_gridspec(12, 8)
fig_ax1 = fig.add_subplot(gs[0:3, :4])
fig_ax1.set_title("First learned shapelet plotted (in red) on top of its 10 best matching time series.")
for i in numpy.argsort(dist_s1)[:1]:#10
    fig_ax1.plot(X_test[i, 0], color='black', alpha=0.5)
    _, pos = torch_dist_ts_shapelet(X_test[i], shapelets[0])
    fig_ax1.plot(lead_pad_shapelet(shapelets[0, 0], pos), color='#F03613', alpha=0.5)


fig_ax2 = fig.add_subplot(gs[0:3, 4:])
fig_ax2.set_title("Second learned shapelet plotted (in red) on top of its 10 best matching time series.")
for i in numpy.argsort(dist_s2)[:1]:#1
    fig_ax2.plot(X_test[i, 0], color='black', alpha=0.5)
    _, pos = torch_dist_ts_shapelet(X_test[i], shapelets[1])
    fig_ax2.plot(lead_pad_shapelet(shapelets[1, 0], pos), color='#F03613', alpha=0.5)

#===
# fig_ax3 = fig.add_subplot(gs[4:, :])
# fig_ax3.set_title("The decision boundaries learned by the model to separate the four classes.")
# color = {0: '#F03613', 1: '#7BD4CC', 2: '#00281F', 3: '#BEA42E'}
# fig_ax3.scatter(dist_s1, dist_s2, color=[color[l] for l in y_test])


# viridis = cm.get_cmap('viridis', 4)
# # Create a meshgrid of the decision boundaries
# xmin = numpy.min(shapelet_transform[:, 0]) - 0.1
# xmax = numpy.max(shapelet_transform[:, 0]) + 0.1
# ymin = numpy.min(shapelet_transform[:, 1]) - 0.1
# ymax = numpy.max(shapelet_transform[:, 1]) + 0.1
# xx, yy = numpy.meshgrid(numpy.arange(xmin, xmax, (xmax - xmin)/200),
#                         numpy.arange(ymin, ymax, (ymax - ymin)/200))
# Z = []
# for x, y in numpy.c_[xx.ravel(), yy.ravel()]:
#     Z.append(numpy.argmax([biases[i] + weights[i][0]*x + weights[i][1]*y
#                            for i in range(3)]))#4
# Z = numpy.array(Z).reshape(xx.shape)
# fig_ax3.contourf(xx, yy, Z / 3, cmap=viridis, alpha=0.25)
# fig_ax3.set_xlabel("$dist(x, s_1)$", fontsize=14)
# fig_ax3.set_ylabel("$dist(x, s_2)$", fontsize=14)

#===
caption = """Shapelets learned for the FaceFour dataset of the UCR archive plotted on top of the best matching time series (top two pictures).
        And the corresponding learned decision boundaries of the linear classifier on top of the shapelet transformed test data (bottom picture)."""
pyplot.figtext(0.5, -0.1, caption, wrap=True, horizontalalignment='center', fontsize=14)
pyplot.savefig('learning_shapelets.png', facecolor=fig.get_facecolor(), bbox_inches="tight")
pyplot.show()
