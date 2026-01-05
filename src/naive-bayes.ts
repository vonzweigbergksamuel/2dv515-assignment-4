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
}
