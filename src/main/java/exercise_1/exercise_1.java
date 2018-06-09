package exercise_1;

import org.apache.hadoop.yarn.webapp.hamlet.HamletSpec;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.LinearSVC;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.*;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.catalyst.plans.logical.statsEstimation.ValueInterval;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.regex.Matcher;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.types.DataTypes.FloatType;
import static org.apache.spark.sql.types.DataTypes.IntegerType;


public class exercise_1 {


    private static Integer mapHemo(Double num) {
        if (num == null) {
            return 0;
        } else if (num < 12) {
            return 1;
        } else {
            return 2;
        }
    }

    private static Integer mapCreatinina(Double num) {
        if (num == null) {
            return 0;
        } else if (num > 1.11) {
            return 1;
        } else {
            return 2;
        }
    }

    private static Integer mapAlbumina(Double num) {
        if (num == null) {
            return 0;
        } else if (num < 3.5 || num > 5) {
            return 1;
        } else {
            return 2;
        }
    }


    private static Integer mapBarthel(Integer num) {
        if (num == null) {
            return 0;
        } else if (num < 20) {
            return 1;
        } else if (num < 61) {
            return 2;
        } else if (num < 91) {
            return 3;
        } else if (num < 99) {
            return 4;
        } else {
            return 5;
        }
    }


    private static Integer mapPfeiffer(Integer num) {
        if (num == null) {
            return 0;
        } else if (num < 2) {
            return 1;
        } else if (num < 4) {
            return 2;
        } else if (num < 8) {
            return 3;
        } else {
            return 4;
        }
    }

    private static Integer mapDifBarthel(Integer num) {
        if (num == null) {
            return 0;
        } else if (num < -20) {
            return 1;
        } else if (num < 20) {
            return 2;
        } else {
            return 3;
        }
    }

    private static Integer mapDifPfeiffer(Integer num) {
        if (num == null) {
            return 0;
        } else if (num < -2) {
            return 1;
        } else if (num < 2) {
            return 2;
        } else {
            return 3;
        }
    }

    public static Dataset<Row> readEstimationFile(SparkSession ss,
                                                  JavaSparkContext jsc, String path) {
        return new DataFrameReader(ss)
                .option("nullValue", "NA")
                .option("inferSchema", "true")
                .option("header", "true")
                .option("sep", ";")
                .csv(path);
    }


    // Transforma el vector de palabras al modelo TF_IDF
    public static Dataset<Row> transformTFIDF(Dataset<Row> ds, int numFeatures) {
        Tokenizer tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words");

        Dataset<Row> wordsMail = tokenizer.transform(ds);

        HashingTF hashingTF = new HashingTF()
                .setInputCol("words")
                .setOutputCol("features")
                .setNumFeatures(numFeatures);

        Dataset<Row> featurizedData = hashingTF.transform(wordsMail);
        return (featurizedData);
    }

    // Ajusta un modelo SVM lineal mediante CV seleccionando el mejor parámetro C
    public static CrossValidatorModel fitModel(Dataset<Row> train) {

        LinearSVC lsvc = new LinearSVC()
                .setMaxIter(5)
                .setLabelCol("label")
                .setFeaturesCol("features");

        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(lsvc.regParam(), new double[]{10.0, 1.0, 0.1})
                .build();
        CrossValidator cv = new CrossValidator()
                .setEstimator(lsvc)
                .setEvaluator(new MulticlassClassificationEvaluator()
                        .setMetricName("accuracy")
                        .setLabelCol("label")
                        .setPredictionCol("prediction"))
                .setEstimatorParamMaps(paramGrid).setNumFolds(5);
        CrossValidatorModel cvModel = cv.fit(train);
        return (cvModel);
    }

    final static String VAR_DIA_ESTANCIA = "DiasEstancia";
    final static String VAR_DIA_ESTANCIA_DISC = "DiasEstanciaDisc";

    final static double[] SPLITS_DIA_ESTANCIA = {Double.NEGATIVE_INFINITY, 12, 41, 69, Double.POSITIVE_INFINITY};

    final static double[] SPLITS_BARTHEL = {Double.NEGATIVE_INFINITY, -999, 20, 61, 91, 99, Double.POSITIVE_INFINITY};

    private final static String[] CATEGORIES_ORGANIC = {"DESCONOCIDO", "ANORMAL", "NORMAL"};

    private static final String HEMO_COL = "Hemoglobina";
    private static final String CREA_COL = "Creatinina";
    private static final String ALBU_COL = "Albumina";
    private static final String BAR_COL = "Barthel";
    private static final String PFF_COL = "Pfeiffer";
    private static final String DIF_BAR_COL = "DiferenciaBarthel";
    private static final String DIF_PFF_COL = "DiferenciaPfeiffer";
    private final static String IND_DEM_COL = "IndicadorDemencia";
    private final static String IND_CONS_COL = "IndicadorConstipacion";
    private final static String IND_SORD_COL = "IndicadorSordera";
    private final static String IND_ALT_VIS_COL = "IndicadorAltVisual";

    private final static String LIST_DIAGNOSTICOS_PRI_COL = "ListaDiagnosticosPri";
    private final static String LIST_DIAGNOSTICOS_SEC_COL = "ListaDiagnosticosSec";
    private final static String LIST_PROCEDIMIENTOS_PRI_COL = "ListaProcedimientosPri";
    private final static String LIST_PROCEDIMIENTOS_SEC_COL = "ListaProcedimientosSec";
    private final static String LIST_CAUSAS_EXTERNAS_COL = "ListaCausasExternas";

    private static final String JOKER_COL = "joker";
    private static final String VALUE_COL = "value";
    private static final String[] INPUT_COLUMS = new String[]{
            HEMO_COL,
            CREA_COL,
            ALBU_COL,
            BAR_COL,
            PFF_COL,
            DIF_BAR_COL,
            DIF_PFF_COL
    };
    private static final String OUTPUT_COL = "output";
    private static final String[] OUTPUT_COLUMNS =  new String[]{
            HEMO_COL + "_OUTPUT",
            CREA_COL + "_OUTPUT",
            ALBU_COL + "_OUTPUT",
            BAR_COL + "_OUTPUT",
            PFF_COL + "_OUTPUT",
            DIF_BAR_COL + "_OUTPUT",
            DIF_PFF_COL + "_OUTPUT"
    };;
    private static final Integer NULLABLE_INDEX = 3;

    private static Dataset<Row> tokenizerAndVectorizerByComma(
            final Dataset<Row> input,
            final String inputCol,
            final String outputCol,
            final String newOutputCol
    ) {
        RegexTokenizer tokenizer = new RegexTokenizer()
                .setInputCol(inputCol)
                .setOutputCol(outputCol)
                .setPattern(",");

        Dataset<Row> output = tokenizer.transform(input);
        Word2Vec word2Vec = new Word2Vec()
                .setInputCol(outputCol)
                .setOutputCol(newOutputCol);
        Word2VecModel word2VecModel = word2Vec.fit(output);
        return word2VecModel.transform(output);
    }

    private static Integer getValueOrDefault(Integer x, Integer value){
        return x == null ? value : x;
    }

    public static void pacientesSim(SparkSession ss) {
        JavaSparkContext jsc = new JavaSparkContext(ss.sparkContext());
        Dataset<Row> estimations = readEstimationFile(ss, jsc, "src/main/resources/PacientesSim.csv");

        // Transformaciones
        // Tr 1
        Bucketizer bucketizer = new Bucketizer()
                .setInputCol(VAR_DIA_ESTANCIA)
                .setOutputCol(VAR_DIA_ESTANCIA_DISC)
                .setSplits(SPLITS_DIA_ESTANCIA);
        Dataset<Row> output = bucketizer.transform(estimations);
        // Tr 2

        ss.udf().register("mapHemo", (Double x) -> mapHemo(x), IntegerType);
        ss.udf().register("mapCreatinina", (Double x) -> mapCreatinina(x), IntegerType);
        ss.udf().register("mapAlbumina", (Double x) -> mapAlbumina(x), IntegerType);
        ss.udf().register("mapBarthel", (Integer x) -> mapBarthel(x), IntegerType);
        ss.udf().register("mapPfeiffer", (Integer x) -> mapPfeiffer(x), IntegerType);
        ss.udf().register("mapDifBarthel", (Integer x) -> mapDifBarthel(x), IntegerType);
        ss.udf().register("mapDifPfeiffer", (Integer x) -> mapDifPfeiffer(x), IntegerType);
        ss.udf().register("valueOrZero", (Integer x) -> getValueOrDefault(x, 0), IntegerType);
        ss.udf().register("valueOrTwo", (Integer x) -> getValueOrDefault(x, 2), IntegerType);

        Dataset<Row> output2 = output.withColumn(HEMO_COL, callUDF("mapHemo", output.col(HEMO_COL)))
                                    .withColumn(CREA_COL, callUDF("mapCreatinina", output.col(CREA_COL)))
                                    .withColumn(ALBU_COL, callUDF("mapAlbumina", output.col(ALBU_COL)))
                                    .withColumn(BAR_COL, callUDF("mapBarthel", output.col(BAR_COL)))
                                    .withColumn(PFF_COL, callUDF("mapPfeiffer", output.col(PFF_COL)))
                                    .withColumn(DIF_BAR_COL, callUDF("mapDifBarthel", output.col(DIF_BAR_COL)))
                                    .withColumn(DIF_PFF_COL, callUDF("mapDifPfeiffer", output.col(DIF_PFF_COL)))
                                    .withColumn(IND_DEM_COL, callUDF("valueOrZero", output.col(IND_DEM_COL)))
                                    .withColumn(IND_CONS_COL, callUDF("valueOrTwo", output.col(IND_CONS_COL)))
                                    .withColumn(IND_SORD_COL, callUDF("valueOrTwo", output.col(IND_SORD_COL)))
                                    .withColumn(IND_ALT_VIS_COL, callUDF("valueOrTwo", output.col(IND_ALT_VIS_COL)));

        // Encoding (Tr 7)
        OneHotEncoderEstimator a = new OneHotEncoderEstimator()
                .setInputCols(INPUT_COLUMS)
                .setOutputCols(OUTPUT_COLUMNS);

        Dataset<Row> output3 = a.fit(output2).transform(output2);
        output3.show();

        // Lista de valores

        Dataset<Row> listDiagnosticosPriRow = tokenizerAndVectorizerByComma(output, LIST_DIAGNOSTICOS_PRI_COL, JOKER_COL, OUTPUT_COL);
        Dataset<Row> listDiagnosticosSecRow = tokenizerAndVectorizerByComma(output, LIST_DIAGNOSTICOS_SEC_COL, JOKER_COL, OUTPUT_COL);
        Dataset<Row> listProcedimientosPriRow = tokenizerAndVectorizerByComma(output, LIST_PROCEDIMIENTOS_PRI_COL, JOKER_COL, OUTPUT_COL);
        Dataset<Row> listProcedimientosSecRow = tokenizerAndVectorizerByComma(output, LIST_PROCEDIMIENTOS_SEC_COL, JOKER_COL, OUTPUT_COL);
        Dataset<Row> listCausasExternasRow = tokenizerAndVectorizerByComma(output, LIST_CAUSAS_EXTERNAS_COL, JOKER_COL, OUTPUT_COL);

//      Dataset<Row> finalOutput = output
//                .union(hemoEncoded.select(OUTPUT_COL));
//                .withColumn(CREA_COL, creaEncoded.select(OUTPUT_COL))
//                .withColumn(ALBU_COL, albuEncoded.select(OUTPUT_COL))
//                .withColumn(BAR_COL, barEncoded.select(OUTPUT_COL))
//                .withColumn(PFF_COL, pfeiEncoded.select(OUTPUT_COL))
//                .withColumn(DIF_BAR_COL, difBarEncoded.select(OUTPUT_COL))
//                .withColumn(DIF_PFF_COL, difPfeiEncoded.select(OUTPUT_COL))
//                .withColumn(IND_DEM_COL, dementiaEndoded.select(OUTPUT_COL))
//                .withColumn(IND_CONS_COL, constipationEncoded.select(OUTPUT_COL))
//                .withColumn(IND_SORD_COL, deafnessEncoded.select(OUTPUT_COL))
//                .withColumn(IND_ALT_VIS_COL, visualDisturbance.select(OUTPUT_COL))
//                .withColumn(LIST_DIAGNOSTICOS_PRI_COL, listDiagnosticosPriRow.select(OUTPUT_COL))
//                .withColumn(LIST_DIAGNOSTICOS_SEC_COL, listDiagnosticosSecRow.select(OUTPUT_COL))
//                .withColumn(LIST_PROCEDIMIENTOS_PRI_COL, listProcedimientosPriRow.select(OUTPUT_COL))
//                .withColumn(LIST_PROCEDIMIENTOS_SEC_COL, listProcedimientosSecRow.select(OUTPUT_COL))
//                .withColumn(LIST_CAUSAS_EXTERNAS_COL, listCausasExternasRow.select(OUTPUT_COL));

//        finalOutput.show();

        /*

        // Dividimos el dataset en train i test
	    Dataset<Row>[] splits= estimations.randomSplit(new double[] {0.3,0.7});
	    Dataset<Row> train = splits[1];
	    Dataset<Row> test = splits[0];

	    // Aseguramos permanencia del train en la memoria de los workers si es posible
	    train.persist();

	    // Ajustamos el modelo (SVM con CV)
	    CrossValidatorModel cvModel = fitModel(train);

	    // Predicciones sobre test set
	    Dataset<Row> predictions = cvModel.transform(test).select("prediction","label");

	    // Definimos un evaluador
	    MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
			        .setMetricName("accuracy")
			        .setLabelCol("label")
			        .setPredictionCol("prediction");

	    double accuracy = evaluator.evaluate(predictions);
	    System.out.println("Train samples: "+train.count());
	    System.out.println("Test samples: "+test.count());
	    System.out.println("Test Error = " + (1 - accuracy));
	    */
        ss.stop();
    }
}
