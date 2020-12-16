from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import auc, confusion_matrix, roc_curve
from typing import List


class ModelingUtils():
	def __init__(self, train_x, clf, test_x, actual, predictions, predicted_probs, pos_class_label, neg_class_label):
		"""
		Class to display model results

		:param train_x: Pandas dataframe containing features as columns
		:param clf: Fitted estimator (ex. random forest model)
		:param test_x: Pandas dataframe containing independent variables for the estimator
		:param actual: Pandas series containing dependent variable for the estimator
		:param predictions: Predicted labels
		:param predicted_probs: Predicted probabilities
		:param pos_class_label: Name of positive class
		:param neg_class_label: Name of negative class
		"""
		self.train_x = train_x
		self.clf = clf
		self.test_x = test_x
		self.actual = actual
		self.predictions = predictions
		self.predicted_probs = predicted_probs
		self.pos_class_label = pos_class_label
		self.neg_class_label = neg_class_label

	def _create_correlation_linkage(self) -> np.ndarray:
		"""
		Helper function to create a linkage matrix encoding the hierarchical clustering.
		"""
		corr = spearmanr(self.train_x).correlation
		return hierarchy.ward(corr)

	def select_features_after_removing_highly_correlated_feature(self, correlation_cutoff=1) -> List[str]:
		"""
		Determines Spearman correlations between features and then clusters the
		features after removing those with a cophenetic distance less than the
		given correlation cutoff.

		:param correlation_cutoff: Minimum cophenetic distance allowed between features in a cluster
		:return: List of feature names after removing highly correlated features
		"""
		corr_linkage = self._create_correlation_linkage()

		cluster_ids = hierarchy.fcluster(
			corr_linkage, correlation_cutoff, criterion="distance"
		)
		cluster_id_to_feature_ids = defaultdict(list)
		for idx, cluster_id in enumerate(cluster_ids):
			cluster_id_to_feature_ids[cluster_id].append(idx)
		selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
		col_names = np.array(self.train_x.columns)
		selected_feature_names = list(col_names[selected_features])

		return selected_feature_names

	def plot_feature_dendrogram(self, figsize=(12, 8)) -> None:
		"""
		Plots a dendrogram showing the hierarchical clustering between features.

		:param figsize: Size of resulting figure
		"""
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
		corr = spearmanr(self.train_x).correlation
		corr_linkage = self._create_correlation_linkage()
		dendro = hierarchy.dendrogram(
			corr_linkage, labels=self.train_x.columns.tolist(), ax=ax1, leaf_rotation=90
		)
		dendro_idx = np.arange(0, len(dendro["ivl"]))

		ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
		ax2.set_xticks(dendro_idx)
		ax2.set_yticks(dendro_idx)
		ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
		ax2.set_yticklabels(dendro["ivl"])
		fig.tight_layout()
		plt.show()

	def plot_permutation_importances(self, figsize=(15, 8)):
		"""
		Plots permutation-based feature importances given a fitted estimator (clf).

		:param figsize: Size of resulting figure
		"""
		result = permutation_importance(
			self.clf, self.test_x, self.actual, n_repeats=10, random_state=42, n_jobs=1
		)
		sorted_idx = result.importances_mean.argsort()

		fig, ax = plt.subplots(figsize=figsize)
		ax.boxplot(
			result.importances[sorted_idx].T, vert=False, labels=self.test_x.columns[sorted_idx]
		)
		ax.set_title("Permutation Importances (test set)")
		fig.tight_layout()
		plt.show()

	def plot_confusion_matrix(self, figsize=(5, 5)) -> None:
		"""
		Plots a confusion matrix.

		:param figsize: Size of resulting figure
		"""
		tn, fp, fn, tp = confusion_matrix(self.actual, self.predictions).ravel()
		matrix = np.array([[tp, fp], [fn, tn]]).T
		label_names = [self.pos_class_label, self.neg_class_label]

		_, ax = plt.subplots(figsize=figsize)
		sns.heatmap(matrix, annot=True, cmap=plt.cm.Blues, fmt="g")
		ax.set_ylim([0, 2])
		plt.xticks([0.5, 1.5], labels=label_names)
		plt.yticks([0.5, 1.5], labels=label_names)
		plt.ylabel("Actual label")
		plt.xlabel("Predicted label")
		plt.show()

	def _prob_plot(self, data: List[float], color: str, label: str, bins: int) -> None:
		"""
		Helper function to plot the distribution of predicted probabilities for a given label.

		:param data: Predicted probabilities
		:param color: Color to plot
		:param label: Name of the class represented in the label
		"""
		_, bins, _ = plt.hist(data, 50, color=color, alpha=0.5, label=label, density=True, bins=bins)
		width = bins[1] - bins[0]
		plt.xlim(0, 1)
		plt.legend()
		plt.xlabel("Predicted Probability of {}".format(self.pos_class_label))
		plt.ylabel("Percent of Observations")

	def plot_probability_distribution(self, bins=50, figsize=(5, 5)) -> None:
		"""
		Plots the distribution of predicted probabilities for a binary classification

		:param bins: Number of bins to plot
		:param figsize: Size of resulting figure
		"""
		predicted = np.array(self.predicted_probs)
		actual = np.array(self.actual)

		_, ax = plt.subplots(figsize=figsize)
		self._prob_plot(
			[x[0] for x in predicted[actual == 1]],
			color="red",
			label="True {}".format(self.pos_class_label),
			bins=bins
		)
		self._prob_plot(
			[x[1] for x in predicted[actual == 0]],
			color="blue",
			label="True {}".format(self.neg_class_label),
			bins=bins
		)
		plt.title("Distribution of Predicted Probabilities")
		plt.show()

	def plot_calibration_curve(self, figsize=(5, 5)) -> None:
		"""
		Plots a reliability curve to assess the calibration of the model.

		:param figsize: Size of resulting figure
		"""
		fraction_of_positives, mean_predicted_value = calibration_curve(
			self.actual, self.predicted_probs[:, 1], n_bins=10
		)
		_, ax = plt.subplots(figsize=figsize)
		plt.plot(mean_predicted_value, fraction_of_positives, label="Your Model")
		plt.plot((0, 1), (0, 1), label="Perfect Calibration")
		plt.ylabel("Fraction of positives")
		plt.xlabel("Mean Prediction")
		plt.ylim([-0.05, 1.05])
		plt.legend(loc="lower right")
		plt.title("Calibration plots  (reliability curve)")
		plt.show()

	def plot_roc_curve(self, figsize=(7, 7)) -> None:
		"""
		Plots a reliability curve to assess the calibration of the model.

		:param figsize: Size of resulting figure
		"""
		fpr, tpr, _ = roc_curve(self.actual, self.predicted_probs[:, 1])
		roc_auc = auc(fpr, tpr)

		plt.figure(figsize=figsize)
		lw = 2
		plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

		plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title(f"Positive Class={self.pos_class_label}")
		plt.legend(loc="lower right")
		plt.show()

