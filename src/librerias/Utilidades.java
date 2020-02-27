package librerias;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Random;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.Bagging;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.ChebyshevDistance;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.ManhattanDistance;
import weka.core.SelectedTag;
import weka.core.SerializationHelper;
import weka.core.Utils;
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

	public Evaluation holdOut(Classifier pClasificador, Instances pTrain, Instances pTest, String pFichero)
			throws Exception {
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
		double bestF = -1.0;
		IBk cls;
		Evaluation eval;
		int minIndex = getFreqMinClassIndex(pData);

		System.out.println("Searching best parameters...\n");

		int kMax = (int) (pData.numInstances() * 0.4);

		for (int k = 1; k <= kMax; k++) {
			for (int w = 1; w <= 3; w++) {
				for (int d = 1; d <= 3; d++) {
					// Manhattan con WEIGHT_SIMILARITY casca, por lo que nos lo saltamos
					if (d != 1 || w != 2) {
						cls = getKNN(pData, k, d, w);
						eval = new Evaluation(pData);
						eval.crossValidateModel(cls, pData, 10, new Random(1));
						if (eval.fMeasure(minIndex) > bestF) {
							bestK = k;
							bestDistance = d;
							bestWeight = w;
							bestF = eval.fMeasure(minIndex);
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
		int numArboles = 1000; // numIterations
		int kMax = Double.valueOf(Math.sqrt(Double.valueOf(pData.numAttributes()))).intValue();

		int bestI = 0;
		int bestK = 0;
		double bestF = -1.0;
		double fAnt = 0;

		RandomForest cls = new RandomForest();
		Evaluation eval;
		int minIndex = getFreqMinClassIndex(pData);

		System.out.println("Searching best parameters...\n");

		for (int i = 1; i <= numArboles; i++) {
			cls.setNumIterations(i);
			for (int k = 1; k <= kMax; k++) {
				cls.setNumFeatures(k); // numFeatures
				cls.buildClassifier(pData);
				eval = new Evaluation(pData);
				eval.crossValidateModel(cls, pData, 10, new Random(1));
				if (eval.fMeasure(minIndex) > bestF) {
					bestK = k;
					bestI = i;
					bestF = eval.fMeasure(minIndex);
				}
			}
			if (i % 100 == 0 && fAnt != 0.0) {
				if (pararBarrido(bestF, fAnt))
					i = numArboles;
				fAnt = bestF;
			} else if (i % 100 == 0 && fAnt == 0) {
				fAnt = bestF;
			}
		}

		cls = new RandomForest();
		cls.setNumIterations(bestI);
		cls.setNumFeatures(bestK);
		System.out.println("################################");
		System.out.println("BEST RANDOM FOREST PARAMETERS:");
		System.out.println("ITERATIONS: " + bestI);
		System.out.println("FEATURES: " + bestK);
		System.out.println("################################");
		return cls;
	}

	private boolean pararBarrido(double fMeasureAct, double fMeasurePre) {
		if (fMeasureAct - fMeasurePre < 0.05)
			return true;
		else
			return false;
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	/* BARRIDO DE PARAMETROS ONE R */

	public OneR configurarOneR(Instances pData) throws Exception {
		int bestB = 0;
		double bestF = -1.0;

		OneR cls;
		Evaluation eval;
		int minIndex = getFreqMinClassIndex(pData);

		System.out.println("Searching best parameters...\n");

		for (int b = 1; b <= pData.numInstances(); b++) {
			cls = new OneR();
			cls.setMinBucketSize(b);
			eval = new Evaluation(pData);
			eval.crossValidateModel(cls, pData, 10, new Random(1));
			double fAct = eval.fMeasure(minIndex);
			if (fAct > bestF) {
				bestB = b;
				bestF = fAct;
			}
		}

		cls = new OneR();
		cls.setMinBucketSize(bestB);
		System.out.println("################################");
		System.out.println("BEST OneR PARAMETERS:");
		System.out.println("BUCKET SIZE: " + bestB);
		System.out.println("################################");
		return cls;
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	/* BARRIDO DE PARAMETROS ZERO R */

	public ZeroR configurarZeroR(Instances pData) throws Exception {
		int bestB = 0;
		double bestF = -1.0;

		ZeroR cls;
		Evaluation eval;
		int minIndex = getFreqMinClassIndex(pData);

		System.out.println("Searching best parameters...\n");

		for (int b = 1; b <= pData.numInstances(); b++) {
			cls = new ZeroR();
			cls.setBatchSize(String.valueOf(b));
			eval = new Evaluation(pData);
			eval.crossValidateModel(cls, pData, 10, new Random(1));
			double fAct = eval.fMeasure(minIndex);
			if (fAct > bestF) {
				bestB = b;
				bestF = fAct;
			}
		}

		cls = new ZeroR();
		cls.setBatchSize(String.valueOf(bestB));
		System.out.println("################################");
		System.out.println("BEST ZeroR PARAMETERS:");
		System.out.println("BATCH SIZE: " + bestB);
		System.out.println("################################");
		return cls;
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	/* BARRIDO DE PARAMETROS BAGGING */

	public Bagging configurarBagging(Instances pData) throws Exception {
		int bestB = 1;
		double bestF = -1.0;

		Bagging cls = new Bagging();
		Evaluation eval;
		int minIndex = getFreqMinClassIndex(pData);

		System.out.println("Searching best parameters...\n");

		for (int b = 10; b <= 100; b++) {
			cls.setBagSizePercent(b);
			cls.buildClassifier(pData);
			eval = new Evaluation(pData);
			eval.crossValidateModel(cls, pData, 10, new Random(1));
			if (eval.fMeasure(minIndex) > bestF) {
				bestB = b;
				bestF = eval.fMeasure(minIndex);
			}
		}

		cls = new Bagging();
		cls.setBagSizePercent(bestB);
		System.out.println("################################");
		System.out.println("BEST BAGGING PARAMETERS:");
		System.out.println("BAG SIZE PERCENT: " + bestB);
		System.out.println("################################");
		return cls;
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	/* BARRIDO DE PARAMETROS J48 */

	public J48 configurarJ48(Instances pData) throws Exception {
		int bestFolds = 1; // Num vueltas
		int bestI = 1;
		double bestF = -1.0;

		int maxI = (int) (pData.numInstances() * 0.4);

		J48 cls = new J48();
		Evaluation eval;
		int minIndex = getFreqMinClassIndex(pData);

		System.out.println("Searching best parameters...\n");

		for (int i = 1; i <= maxI; i++) {
			cls.setMinNumObj(i);
			for (int f = 1; f <= 10; f++) {
				cls.setNumFolds(f);
				cls.buildClassifier(pData);
				eval = new Evaluation(pData);
				eval.crossValidateModel(cls, pData, 10, new Random(1));
				if (eval.fMeasure(minIndex) > bestF) {
					bestFolds = f;
					bestI = i;
					bestF = eval.fMeasure(minIndex);
				}
			}
		}

		cls = new J48();
		cls.setNumFolds(bestFolds);
		cls.setMinNumObj(bestI);
		System.out.println("################################");
		System.out.println("BEST J48 PARAMETERS:");
		System.out.println("MIN NUM OBJECTS: " + bestI);
		System.out.println("NUM FOLDS: " + bestFolds);
		System.out.println("################################");
		return cls;
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	/* BARRIDO DE PARAMETROS SMO */

	public SMO configurarSMO(Instances pData) throws Exception {
		int bestE = 0;
		double bestF = -1.0;

		SMO cls = new SMO();
		Evaluation eval;
		PolyKernel kernel = new PolyKernel();
		int minIndex = getFreqMinClassIndex(pData);

		System.out.println("Searching best parameters...\n");

		for (int e = 0; e <= 5; e++) {
			kernel.setExponent(e);
			cls.setKernel(kernel);
			cls.buildClassifier(pData);
			eval = new Evaluation(pData);
			eval.crossValidateModel(cls, pData, 10, new Random(1));
			if (eval.fMeasure(minIndex) > bestF) {
				bestE = e;
				bestF = eval.fMeasure(minIndex);
			}
		}

		cls = new SMO();
		kernel.setExponent(bestE);
		cls.setKernel(kernel);
		System.out.println("################################");
		System.out.println("BEST SMO PARAMETERS:");
		System.out.println("KERNEL EXPONENT: " + bestE);
		System.out.println("################################");
		return cls;
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////

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

	private int getFreqMinClassIndex(Instances pData) {
		int index = Utils.minIndex(pData.attributeStats(pData.classIndex()).nominalCounts);

		return index;
	}

}
