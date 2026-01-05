interface ClassModel {
	name: string;
	prior: number;
	mean: number[];
	variance: number[];
}

export class NaiveBayes {
	private model: ClassModel[] = [];

	fit(X: number[][], y: string[]): void {
		if (X.length === 0 || !X[0]) return;

		const classes = new Set(y);

		for (const className of classes) {
			let classCount = 0;
			const featureSums: number[] = new Array(X[0].length).fill(0);

			for (let i = 0; i < X.length; i++) {
				if (y[i] === className) {
					classCount++;
					const currentRow = X[i];
					if (currentRow) {
						for (let j = 0; j < currentRow.length; j++) {
							const value = currentRow[j];
							const sum = featureSums[j];
							if (value !== undefined && sum !== undefined) {
								featureSums[j] = sum + value;
							}
						}
					}
				}
			}

			if (classCount > 0) {
				const featureMeans = featureSums.map((sum) => sum / classCount);
				const prior = classCount / y.length;
				const variance = this.calculateVariance(className, featureMeans, X, y);

				this.model.push({
					name: className,
					prior,
					mean: featureMeans,
					variance,
				});
			}
		}
	}

	private calculateVariance(
		className: string,
		means: number[],
		X: number[][],
		y: string[],
	): number[] {
		let classCount = 0;
		const varianceSums: number[] = new Array(means.length).fill(0);

		for (let i = 0; i < X.length; i++) {
			if (y[i] === className) {
				classCount++;
				const currentRow = X[i];
				if (currentRow) {
					for (let j = 0; j < currentRow.length; j++) {
						const value = currentRow[j];
						const mean = means[j];
						const varSum = varianceSums[j];
						if (
							value !== undefined &&
							mean !== undefined &&
							varSum !== undefined
						) {
							const diff = value - mean;
							varianceSums[j] = varSum + diff ** 2;
						}
					}
				}
			}
		}

		return varianceSums.map((sum) => sum / classCount);
	}

	predict(X: number[][]): string[] {
		const predictions: string[] = [];

		for (const sample of X) {
			let bestClass: string | null = null;
			let bestScore = -1;

			for (const classModel of this.model) {
				let score = classModel.prior;

				for (let i = 0; i < sample.length; i++) {
					let variance = classModel.variance[i];
					const sampleValue = sample[i];
					const meanValue = classModel.mean[i];

					if (variance === undefined || variance === 0) {
						variance = 1e-9;
					}

					if (sampleValue !== undefined && meanValue !== undefined) {
						score *= this.calculateProbability(
							sampleValue,
							meanValue,
							variance,
						);
					}
				}

				if (score > bestScore) {
					bestScore = score;
					bestClass = classModel.name;
				}
			}

			if (bestClass !== null) {
				predictions.push(bestClass);
			}
		}

		return predictions;
	}

	private calculateProbability(
		x: number,
		mean: number,
		variance: number,
	): number {
		const std = Math.sqrt(variance);
		const exponent = Math.exp(-((x - mean) ** 2) / (2 * variance));
		return (1 / (Math.sqrt(2 * Math.PI) * std)) * exponent;
	}

	accuracyScore(predictions: string[], y: string[]): number {
		let correct = 0;

		for (let i = 0; i < predictions.length; i++) {
			if (predictions[i] === y[i]) {
				correct++;
			}
		}

		return correct / predictions.length;
	}

	confusionMatrix(predictions: string[], y: string[]): number[][] {
		const labels = Array.from(new Set(y)).sort();
		const labelToIndex = new Map<string, number>();
		labels.forEach((label, i) => {
			labelToIndex.set(label, i);
		});

		const matrix: number[][] = Array(labels.length)
			.fill(0)
			.map(() => Array(labels.length).fill(0));

		for (let i = 0; i < predictions.length; i++) {
			const trueLabel = y[i];
			const predLabel = predictions[i];

			if (trueLabel && predLabel) {
				const trueIndex = labelToIndex.get(trueLabel);
				const predIndex = labelToIndex.get(predLabel);

				if (trueIndex !== undefined && predIndex !== undefined) {
					const trueRow = matrix[trueIndex];
					if (trueRow) {
						const currentValue = trueRow[predIndex];
						if (currentValue !== undefined) {
							trueRow[predIndex] = currentValue + 1;
						}
					}
				}
			}
		}

		return matrix;
	}
}
