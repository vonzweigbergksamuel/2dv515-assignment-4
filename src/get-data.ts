import { resolve } from "node:path";
import { parseFile } from "fast-csv";

interface DataResult {
	X: number[][];
	y: string[];
}

export async function getData(fileName: string): Promise<DataResult> {
	const X: number[][] = [];
	const y: string[] = [];

	const dataPath = resolve(__dirname, "../data", fileName);

	return new Promise((resolvePromise, reject) => {
		parseFile(dataPath, { headers: true })
			.on("error", (error) => reject(error))
			.on("data", (row: Record<string, string>) => {
				const keys = Object.keys(row);
				const lastKey = keys[keys.length - 1];

				if (!lastKey) return;

				const features = keys
					.slice(0, -1)
					.map((key) => parseFloat(row[key] || "0"));
				const label = row[lastKey] || "";

				X.push(features);
				y.push(label);
			})
			.on("end", () => {
				const combined = X.map((features, i) => ({
					features,
					label: y[i] || "",
				}));

				for (let i = combined.length - 1; i > 0; i--) {
					const j = Math.floor(Math.random() * (i + 1));
					const temp = combined[i];
					const swap = combined[j];
					if (temp && swap) {
						combined[i] = swap;
						combined[j] = temp;
					}
				}

				const shuffledX = combined.map((item) => item.features);
				const shuffledY = combined.map((item) => item.label);

				resolvePromise({ X: shuffledX, y: shuffledY });
			});
	});
}
