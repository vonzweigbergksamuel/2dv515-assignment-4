import { getData } from "@/get-data";
import { NaiveBayes } from "@/naive-bayes";

async function runIris() {
	const { X, y } = await getData("iris.csv");

	const naiveBayes = new NaiveBayes();
	naiveBayes.fit(X, y);
	const predictions = naiveBayes.predict(X);
	const accuracy = naiveBayes.accuracyScore(predictions, y);
	const matrix = naiveBayes.confusionMatrix(predictions, y);

	printResults("Iris", predictions, accuracy, matrix);
}

async function runBanknote() {
	const { X, y } = await getData("banknote_authentication.csv");

	const naiveBayes = new NaiveBayes();
	naiveBayes.fit(X, y);
	const predictions = naiveBayes.predict(X);
	const accuracy = naiveBayes.accuracyScore(predictions, y);
	const matrix = naiveBayes.confusionMatrix(predictions, y);

	printResults("Banknote", predictions, accuracy, matrix);
}

function printResults(
	name: string,
	predictions: string[],
	accuracy: number,
	matrix: number[][],
) {
	console.log(`================== ${name} ==================`);
	console.log();
	console.log(
		`Predictions: [${predictions.slice(0, 10).join(", ")}${predictions.length > 10 ? ", ..." : ""}]`,
	);
	console.log();
	console.log(`Accuracy Score: ${accuracy.toFixed(4)}`);
	console.log();
	console.log("Confusion Matrix:");
	console.log(matrix.map((row) => `[${row.join(", ")}]`).join(",\n"));
	console.log();
}

async function main() {
	await runIris();
	await runBanknote();
}

main();
