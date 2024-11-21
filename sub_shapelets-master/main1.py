#==========
loss_func = nn.CrossEntropyLoss()
# Learning Shapelets as of the paper 'Learning Time Series Shapelets', Grabocka et al. (2014)
learning_shapelets = LearningShapelets(shapelets_size_and_len=shapelets_size_and_len,
                                       in_channels=n_channels,
                                       num_classes=num_classes,
                                       loss_func=loss_func,
                                       to_cuda=True,
                                       verbose=1,
                                       dist_measure=dist_measure)
# OR Learning Shapelets with regularized loss
learning_shapelets = LearningShapelets(shapelets_size_and_len=shapelets_size_and_len,
                                       in_channels=n_channels,
                                       num_classes=num_classes,
                                       loss_func=loss_func,
                                       to_cuda=True,
                                       verbose=1,
                                       dist_measure=dist_measure,
                                       l1=l1,
                                       l2=l2,
                                       k=k)

# (Optionally) Initialize shapelet weights, the original paper uses k-Means
# otherwise the shapelets will be initialized randomly.
# Note: This implementation does not provide an initialization strategy other
# than random initialization.
learning_shapelets.set_shapelet_weights(weights)

# Initialize an optimizer.
# Make sure to first set the weights otherwise the model parameters will have changed.
optimizer = optim.Adam(learning_shapelets.model.parameters(),
                       lr=lr,
                       weight_decay=wd)
learning_shapelets.set_optimizer(optimizer)

# Train model.
losses = learning_shapelets.fit(X_train,
                                y_train,
                                epochs=2000,
                                batch_size=256,
                                shuffle=False,
                                drop_last=False)

# Extract the learned shapelets.
shapelets = learning_shapelets.get_shapelets()

# Shapelet transform a dataset.
shapelet_transform = learning_shapelets.transform(X_test)

# Make predictions.
predictions = learning_shapelets.predict(X_test)