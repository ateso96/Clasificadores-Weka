package dataMining;

import java.util.Random;

import librerias.Utilidades;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.SerializationHelper;

//arg 1: Path del fichero de datos
//arg 2: /Path para guardar el txt con los resultados.
//arg 3: Path para guardar el modelo (.model).
//arg 4: Path para guardar el txt con las predicciones.

public class Main {
	
	private static double porcentajeSplit = 25.0;

	public static void main(String[] args) throws Exception {

		if (args.length != 4)
			System.out.println("ERROR: Se necesitan 4 argumentos de entrada: " 
					+ "\n\t*1: Path del fichero de datos."
					+ "\n\t*2: Path para guardar el txt con los resultados."
					+ "\n\t*3: Path para guardar el modelo (.model)."
					+ "\n\t*4: Path para guardar el txt con las predicciones.");
		else {
			/* 1. Cargar los datos desde el arff */
			Instances data = Utilidades.getUtilidades().cargarDatos(args[0]);

			/* 2. Seleccionar la clase, randomizar y, si fuera necesario, filtrar */
			if (data.attribute("class") == null)
				data.setClassIndex(data.numAttributes() - 1);
			else
				data.setClass(data.attribute("class"));
			Instances newData = Utilidades.getUtilidades().filtrar(data);
			
			newData.randomize(new Random(1));
			int insTrain = (int) (newData.size() * (porcentajeSplit / 100)); // Número de instancias para el train
			int insTest = newData.size() - insTrain; // Número de instancias para el test
			Instances train = new Instances(newData, 0, insTrain - 1);
			Instances test = new Instances(newData, insTrain, insTest - 1);

			/* 3. Inicializar clasificador kNN */
			IBk clasificador = Utilidades.getUtilidades().configurarIBk(newData);
			
			/* 4. Hacer la evaluación */
			/* 4.1. CroosValidation */
			Evaluation evaluator = Utilidades.getUtilidades().crossValidation(clasificador, newData, 10, args[2]);
			
			/* 4.2. Hold-out */
//			Ficheros.getFicheros().guardarResultados(test.toString(), "test.arff");
//			Evaluation evaluator = Utilidades.getUtilidades().holdOut(clasificador, train, test, args[2]);
			
			/* 4.3. No Honesta*/
//			newData.randomize(new Random(1));
//			Evaluation evaluator = Utilidades.getUtilidades().noHonesta(clasificador, newData, args[2]);
			

			/* 5. Mostrar y guardar resultados */
			String resultados = "\n=== Summary ===\n" + evaluator.toSummaryString() + "\n" + "Recall Class 0: "
					+ evaluator.recall(0) + "\n" + "Weighted Average Recall: " + evaluator.weightedRecall() + "\n"
					+ "\n" + evaluator.toClassDetailsString() + "\n" + evaluator.toMatrixString();
			System.out.println(resultados);
			Utilidades.getUtilidades().guardarResultados(resultados, args[1]);

			/* 6. Cargar modelo y hacer predicciones */
			IBk clasificadorModelo = (IBk) SerializationHelper.read(args[2]);
			String predicciones = "=== Predicctions ===\n \ninst# --> instance atributes --> predicted" + "\n";
			Instances dataPred = Utilidades.getUtilidades().filtrar(data); // Para que este filtrado
			for (int i = 0; i < dataPred.numInstances(); i++) {
				double pred = clasificadorModelo.classifyInstance(dataPred.instance(i));
				predicciones += "\n" + i + " --> " + data.instance(i);
				predicciones += " --> " + dataPred.classAttribute().value((int) pred);
			}
			System.out.println(predicciones);
			Utilidades.getUtilidades().guardarResultados(predicciones, args[3]);
		}
	}
}
