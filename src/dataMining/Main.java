package dataMining;

import java.util.Random;

import librerias.Utilidades;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.KStar;
import weka.classifiers.meta.Bagging;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.PART;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instances;
import weka.core.SerializationHelper;

//arg 1: Path del fichero de datos
//arg 2: /Path para guardar el txt con los resultados.
//arg 3: Path para guardar el modelo (.model).
//arg 4: Path para guardar el txt con las predicciones.

public class Main {
	
	private static double porcentajeSplit = 55.0;

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
			
			/* Obtener train y test */
//			newData.randomize(new Random(1));
//			int insTrain = (int) (newData.size() * (porcentajeSplit / 100)); // Número de instancias para el train
//			int insTest = newData.size() - insTrain; // Número de instancias para el test
//			Instances train = new Instances(newData, 0, insTrain - 1);
//			Instances test = new Instances(newData, insTrain, insTest - 1);

			/* Clasificadores */
			/* 3.1. Inicializar clasificador kNN */
			//IBk clasificador = Utilidades.getUtilidades().configurarIBk(newData);
			
			/* 3.2. Inicializar clasificador Random Forest */
			//RandomForest clasificador = Utilidades.getUtilidades().configurarRandomForest(newData);
			
			/* 3.3. Inicializar clasificador OneR */
			//OneR clasificador = Utilidades.getUtilidades().configurarOneR(newData);
			
			/* 3.4. Inicializar clasificador ZeroR */
			//ZeroR clasificador = Utilidades.getUtilidades().configurarZeroR(newData);
			
			/* 3.5. Inicializar clasificador Bagging */
			//Bagging clasificador = Utilidades.getUtilidades().configurarBagging(newData);
			
			/* 3.6. Inicializar clasificador J48 */
			//J48 clasificador = Utilidades.getUtilidades().configurarJ48(newData);
			
			/* 3.6. Inicializar clasificador SMO */
			//SMO clasificador = Utilidades.getUtilidades().configurarSMO(newData);
			
			/* 3.7. Inicializar clasificador PART */
			//PART clasificador = Utilidades.getUtilidades().configurarPART(newData);
			
			/* 3.8. Inicializar clasificador KStar */
			//KStar clasificador = Utilidades.getUtilidades().configurarKStar(newData);
			
			/* 3.9. Inicializar clasificador Random Tree */
			RandomTree clasificador = Utilidades.getUtilidades().configurarRandomTree(newData);
			
			/* 4. Hacer la evaluación */
			/* 4.1. CrossValidation */
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
			SMO clasificadorModelo = (SMO) SerializationHelper.read(args[2]);
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
