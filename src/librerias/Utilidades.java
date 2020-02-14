package librerias;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Random;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.ChebyshevDistance;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.ManhattanDistance;
import weka.core.SelectedTag;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.neighboursearch.LinearNNSearch;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class Utilidades {

	private static Utilidades mUtilidades;

	public static Utilidades getUtilidades() {
		if (mUtilidades == null)
			mUtilidades = new Utilidades();
		return mUtilidades;
	}
	
	/* METODOS PARA EL PREPROCESADO DE DATOS */

	public Instances cargarDatos(String pFichero) throws Exception {

		// Leemos el fichero
		DataSource source = null;
		try {
			source = new DataSource(pFichero);
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("ERROR: el archivo no se ha encontrado");
		}

		// Cargamos las instancias
		Instances data = source.getDataSet();

		return data;
	}

	public void guardarResultados(String pResultado, String pFichero) {
		try {
			PrintWriter out = new PrintWriter(pFichero);
			out.println(pResultado);
			out.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	public Instances filtrar(Instances pData) throws Exception {
		AttributeSelection filter = new AttributeSelection();
		CfsSubsetEval eval = new CfsSubsetEval();
		BestFirst search = new BestFirst();

		filter.setEvaluator(eval);
		filter.setSearch(search);
		filter.setInputFormat(pData);
		Instances newData = Filter.useFilter(pData, filter);

		return newData;
	}
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/* METODOS PARA REALIZAR LAS DIFERENTES EVALUACIONES */

	public Evaluation crossValidation(Classifier pClasificador, Instances pData, int pFolds, String pFichero)
			throws Exception {
		pClasificador.buildClassifier(pData);
		SerializationHelper.write(pFichero, pClasificador);
		Evaluation evaluator = new Evaluation(pData);
		evaluator.crossValidateModel(pClasificador, pData, pFolds, new Random(1));
		return evaluator;
	}

	public Evaluation holdOut(Classifier pClasificador, Instances pTrain, Instances pTest, String pFichero) throws Exception {
		pClasificador.buildClassifier(pTrain);
		SerializationHelper.write(pFichero, pClasificador);
		Evaluation evaluation = new Evaluation(pTrain);
		evaluation.evaluateModel(pClasificador, pTest);
		return evaluation;
	}

	public Evaluation noHonesta(Classifier pClasificador, Instances pData, String pFichero) throws Exception {
		pClasificador.buildClassifier(pData);
		SerializationHelper.write(pFichero, pClasificador);
		Evaluation evaluation = new Evaluation(pData);
		evaluation.evaluateModel(pClasificador, pData);
		return evaluation;
	}
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/* BARRIDO DE PARAMETROS KNN */

	public IBk configurarIBk(Instances pData) throws Exception {
		int bestK = 1;
		int bestDistance = 1;
		int bestWeight = 1;
		double bestF = 0;
		int minClassIndex = getIndClaseMinoritaria(pData);
		IBk cls;
		Evaluation eval;
		
		System.out.println("Searching best parameters...\n");

		for (int k = 1; k <= pData.numInstances() * 0.4; k++) {
			for (int w = 1; w <= 3; w++) {
				for (int d = 1; d <= 3; d++) {
					// Manhattan con WEIGHT_SIMILARITY casca, por lo que nos lo saltamos
					if (d != 1 || w != 2) {
						cls = getKNN(pData, k, d, w);
						eval = new Evaluation(pData);
						eval.crossValidateModel(cls, pData, 10, new Random(1));
						if (eval.fMeasure(minClassIndex) > bestF) {
							bestK = k;
							bestDistance = d;
							bestWeight = w;
							bestF = eval.fMeasure(minClassIndex);
						}
					}
				}
			}
		}
		
		cls = getKNN(pData, bestK, bestDistance, bestWeight);
		System.out.println("################################");
		System.out.println("BEST IBk PARAMETERS:");
		System.out.println("KNN: " + bestK);
		switch (bestDistance) {
		case 1:
			System.out.println("Distance function: Manhattan Distance");
			break;
		case 2:
			System.out.println("Distance function: Euclidean Distance");
			break;
		case 3:
			System.out.println("Distance function: Chebyshev Distance");
			break;
		default:
			break;
		}
		switch (bestWeight) {
		case 1:
			System.out.println("Distance weighting: Weight by 1/distance");
			break;
		case 2:
			System.out.println("Distance weighting: No distance weighting");
			break;
		case 3:
			System.out.println("Distance weighting: Weight by 1 - distance");
			break;
		default:
			break;
		}
		System.out.println("################################");
		return cls;
	}

	private IBk getKNN(Instances data, int k, int distance, int weight) throws Exception {

		IBk estimador = new IBk(k);
		LinearNNSearch search = new LinearNNSearch();

		switch (distance) {
		case 1:
			search.setDistanceFunction(new ManhattanDistance());
			break;
		case 2:
			search.setDistanceFunction(new EuclideanDistance());
			break;
		case 3:
			search.setDistanceFunction(new ChebyshevDistance());
			break;
		}

		switch (weight) {
		case 1:
			estimador.setDistanceWeighting(new SelectedTag(IBk.WEIGHT_INVERSE, IBk.TAGS_WEIGHTING));
			break;
		case 2:
			estimador.setDistanceWeighting(new SelectedTag(IBk.WEIGHT_SIMILARITY, IBk.TAGS_WEIGHTING));
			break;
		case 3:
			estimador.setDistanceWeighting(new SelectedTag(IBk.WEIGHT_NONE, IBk.TAGS_WEIGHTING));
			break;
		}
		return estimador;
	}
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/* BARRIDO DE PARAMETROS RANDOM FOREST */
	
	public RandomForest configurarRandomForest(Instances pData) throws Exception {
		int bestK = 1;
		int bestDistance = 1;
		int bestWeight = 1;
		double bestF = 0;
		int minClassIndex = getIndClaseMinoritaria(pData);
		RandomForest cls;
		Evaluation eval;
		
		System.out.println("Searching best parameters...\n");

		for (int k = 1; k <= pData.numInstances() * 0.4; k++) {
			for (int w = 1; w <= 3; w++) {
				for (int d = 1; d <= 3; d++) {
					// Manhattan con WEIGHT_SIMILARITY casca, por lo que nos lo saltamos
					if (d != 1 || w != 2) {
						cls = getKNN(pData, k, d, w);
						eval = new Evaluation(pData);
						eval.crossValidateModel(cls, pData, 10, new Random(1));
						if (eval.fMeasure(minClassIndex) > bestF) {
							bestK = k;
							bestDistance = d;
							bestWeight = w;
							bestF = eval.fMeasure(minClassIndex);
						}
					}
				}
			}
		}
		
		cls = getKNN(pData, bestK, bestDistance, bestWeight);
		System.out.println("################################");
		System.out.println("BEST IBk PARAMETERS:");
		System.out.println("KNN: " + bestK);
		switch (bestDistance) {
		case 1:
			System.out.println("Distance function: Manhattan Distance");
			break;
		case 2:
			System.out.println("Distance function: Euclidean Distance");
			break;
		case 3:
			System.out.println("Distance function: Chebyshev Distance");
			break;
		default:
			break;
		}
		switch (bestWeight) {
		case 1:
			System.out.println("Distance weighting: Weight by 1/distance");
			break;
		case 2:
			System.out.println("Distance weighting: No distance weighting");
			break;
		case 3:
			System.out.println("Distance weighting: Weight by 1 - distance");
			break;
		default:
			break;
		}
		System.out.println("################################");
		return cls;
	}
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	private int getIndClaseMinoritaria(Instances pData) {
		int ind = pData.classIndex();
		AttributeStats stats = pData.attributeStats(ind);
		int[] frecuencias = stats.nominalCounts;
		int frecMin = frecuencias[0];
		int index = 0;
		int res = 0;
		for (int f: frecuencias) {
			if (f < frecMin && f != 0) {
				frecMin = f;
				res = index;
			}
			index++;
		}
		return res;
	}
	
	private double getFMeasure(Evaluation pEval, Instances pData) {
		Attribute classA = pData.classAttribute();

		// Lista con las fMeasures de cada clase
		double[] fMeasures = new double[classA.numValues()];
		for (int i = 0; i < classA.numValues(); i++) {
			fMeasures[i] = pEval.fMeasure(i);
			if (Double.isNaN(fMeasures[i])) {
				// Todo es 0
				fMeasures[i] = 0;
			}
		}

		// Lista con el peso de cada clase
		int[] weights = new int[classA.numValues()];
		for (int j = 0; j < pData.numInstances(); j++)
			weights[(int) pData.instance(j).classValue()]++;

		// fMeasure es la media del array
		int fMeasure = 0;
		for (int k = 0; k < classA.numValues(); k++)
			fMeasure += weights[k] * fMeasures[k];

		fMeasure = fMeasure / pData.numInstances();
		return fMeasure;
	}

}
