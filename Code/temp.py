# def main():
    #     start_time = time()
#     df = load_data()
#     df = data_cleaning(df)
#     df = feature_engineering(df)
#     # df.to_csv('df.csv', sep=',', index=False)
#     under_sampled_df = under_sampling(df)
    
#     # Define train_data here or retrieve it from under_sampled_df
#     train_data = under_sampled_df.drop('flagged', axis=1)
    
#     rf = RandomForestClassifier(random_state=42, criterion='entropy', max_depth=14, max_features=train_data.shape[1], n_estimators=500)  # Using train_data's shape
    
#     nb = GaussianNB()

#     semi_supervised_learning(under_sampled_df, model=rf, threshold=0.7, iterations=15, algorithm='Random Forest')
#     semi_supervised_learning(under_sampled_df, model=nb, threshold=0.7, iterations=15, algorithm='Naive Bayes')
#     end_time = time()
#     print("Time taken : ", end_time - start_time)


# if __name__ == '__main__':  
#     main()




---new file----
# def main():
    #     start_time = time()
#     df = load_data()
#     df = data_cleaning(df)
#     df = feature_engineering(df)
#     # df.to_csv('df.csv', sep=',', index=False)
#     under_sampled_df = under_sampling(df)
    
#     # Define train_data here or retrieve it from under_sampled_df
#     train_data = under_sampled_df.drop('flagged', axis=1)
    
#     rf = RandomForestClassifier(random_state=42, criterion='entropy', max_depth=14, max_features=train_data.shape[1], n_estimators=500)  # Using train_data's shape
    
#     nb = GaussianNB()

#     semi_supervised_learning(under_sampled_df, model=rf, threshold=0.7, iterations=15, algorithm='Random Forest')
#     semi_supervised_learning(under_sampled_df, model=nb, threshold=0.7, iterations=15, algorithm='Naive Bayes')
#     end_time = time()
#     print("Time taken : ", end_time - start_time)


# if __name__ == '__main__':  
#     main()
 

__new file __
# def plot_confusion_matrix(y_true, y_pred, classes, title=None, cmap=plt.cm.Blues):
#     # Compute confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
#     # Only use the labels that appear in the data

#     fig, ax = plt.subplots()
#     im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#     ax.figure.colorbar(im, ax=ax)
#     # We want to show all ticks...
#     ax.set(xticks=np.arange(cm.shape[1]),
#            yticks=np.arange(cm.shape[0]),
#            xticklabels=classes,
#            yticklabels=classes,
#            title=title,
#            ylabel='True label',
#            xlabel='Predicted label')

#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")

#     # Loop over data dimensions and create text annotations.
#     fmt = 'd'
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, format(cm[i, j], fmt),
#                     ha="center", va="center",
#                     color="white" if cm[i, j] > thresh else "black")
#     fig.tight_layout()

#     return plt