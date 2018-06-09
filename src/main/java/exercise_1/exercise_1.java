package exercise_1;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.DataFrameReader;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.types.DataTypes.IntegerType;


public class exercise_1 {


    private static final String FEATURES_COL = "features";
    public static final String PREDICTION_COL = "prediction";
    public static final String ACCURACY_COL = "accuracy";

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


    // Ajusta un modelo SVM lineal mediante CV seleccionando el mejor parámetro C
    public static CrossValidatorModel fitModel(Dataset<Row> train) {

        RandomForestClassifier rfc = new RandomForestClassifier()
                .setLabelCol(VAR_DIAS_ESTANCIA_DISC)
                .setFeaturesCol(FEATURES_COL);

        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(rfc.numTrees(), new int[]{1, 10, 100})
                .addGrid(rfc.maxDepth(), new int[]{5, 10, 15})
                .build();

        CrossValidator cv = new CrossValidator()
                .setEstimator(rfc)
                .setEvaluator(new MulticlassClassificationEvaluator()
                        .setMetricName(ACCURACY_COL)
                        .setLabelCol(VAR_DIAS_ESTANCIA_DISC)
                        .setPredictionCol(PREDICTION_COL))
                .setEstimatorParamMaps(paramGrid).setNumFolds(5);
        CrossValidatorModel cvModel = cv.fit(train);
        System.out.println(cvModel.bestModel().toString());
        return (cvModel);
    }

    final static String VAR_DIA_ESTANCIA = "DiasEstancia";
    final static String VAR_DIAS_ESTANCIA_DISC = "DiasEstanciaDisc";

    final static double[] SPLITS_DIA_ESTANCIA = {Double.NEGATIVE_INFINITY, 12, 41, 69, Double.POSITIVE_INFINITY};

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
    private final static String[] LIST_ARRAY = new String[]{
            LIST_DIAGNOSTICOS_PRI_COL,
            LIST_DIAGNOSTICOS_SEC_COL,
            LIST_PROCEDIMIENTOS_PRI_COL,
            LIST_PROCEDIMIENTOS_SEC_COL,
            LIST_CAUSAS_EXTERNAS_COL
    };

    private final static String[] LIST_ARRAY_JOKER = new String[]{
            LIST_DIAGNOSTICOS_PRI_COL + "_JOKER",
            LIST_DIAGNOSTICOS_SEC_COL + "_JOKER",
            LIST_PROCEDIMIENTOS_PRI_COL + "_JOKER",
            LIST_PROCEDIMIENTOS_SEC_COL + "_JOKER",
            LIST_CAUSAS_EXTERNAS_COL + "_JOKER"
    };

    private final static String[] LIST_ARRAY_OUTPUT = new String[]{
            LIST_DIAGNOSTICOS_PRI_COL + "_OUTPUT",
            LIST_DIAGNOSTICOS_SEC_COL + "_OUTPUT",
            LIST_PROCEDIMIENTOS_PRI_COL + "_OUTPUT",
            LIST_PROCEDIMIENTOS_SEC_COL + "_OUTPUT",
            LIST_CAUSAS_EXTERNAS_COL + "_OUTPUT"
    };

    private static final String[] INPUT_COLUMS = new String[]{
            HEMO_COL,
            CREA_COL,
            ALBU_COL,
            BAR_COL,
            PFF_COL,
            DIF_BAR_COL,
            DIF_PFF_COL
    };

    private static final String[] OUTPUT_COLUMNS = new String[]{
            HEMO_COL + "_OUTPUT",
            CREA_COL + "_OUTPUT",
            ALBU_COL + "_OUTPUT",
            BAR_COL + "_OUTPUT",
            PFF_COL + "_OUTPUT",
            DIF_BAR_COL + "_OUTPUT",
            DIF_PFF_COL + "_OUTPUT"
    };
    ;

    private static Dataset<Row> tokenizerAndVectorizerByComma(
            Dataset<Row> input,
            final String[] inputCol,
            final String[] jokerCol,
            final String[] outputCol
    ) {

        final Integer inputLength = inputCol.length;
        for (int i = 0; i < inputLength; ++i) {
            RegexTokenizer tokenizer = new RegexTokenizer()
                    .setInputCol(inputCol[i])
                    .setOutputCol(jokerCol[i])
                    .setPattern(",");
            Dataset<Row> output = tokenizer.transform(input);
            Word2Vec word2Vec = new Word2Vec()
                    .setInputCol(jokerCol[i])
                    .setOutputCol(outputCol[i]);
            Word2VecModel word2VecModel = word2Vec.fit(output);
            input = word2VecModel.transform(output);
        }
        return input;
    }

    private static Integer getValueOrDefault(Integer x, Integer value) {
        return x == null ? value : x;
    }

    public static void pacientesSim(SparkSession ss) {
        JavaSparkContext jsc = new JavaSparkContext(ss.sparkContext());
        Dataset<Row> estimations = readEstimationFile(ss, jsc, "src/main/resources/PacientesSim.csv");

        // Tr 1
        Bucketizer bucketizer = new Bucketizer()
                .setInputCol(VAR_DIA_ESTANCIA)
                .setOutputCol(VAR_DIAS_ESTANCIA_DISC)
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

        Dataset<Row> df = tokenizerAndVectorizerByComma(output3,
                LIST_ARRAY,
                LIST_ARRAY_JOKER,
                LIST_ARRAY_OUTPUT);

        final String[] inputColsAssembler = new String[]{
                IND_DEM_COL,
                IND_CONS_COL,
                IND_SORD_COL,
                IND_ALT_VIS_COL,
                LIST_ARRAY_OUTPUT[0],
                LIST_ARRAY_OUTPUT[1],
                LIST_ARRAY_OUTPUT[2],
                LIST_ARRAY_OUTPUT[3],
                LIST_ARRAY_OUTPUT[4],
                OUTPUT_COLUMNS[0],
                OUTPUT_COLUMNS[1],
                OUTPUT_COLUMNS[2],
                OUTPUT_COLUMNS[3],
                OUTPUT_COLUMNS[4],
                OUTPUT_COLUMNS[5],
                OUTPUT_COLUMNS[6]
        };

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(inputColsAssembler)
                .setOutputCol(FEATURES_COL);
        Dataset<Row> assembled = assembler.transform(df);

        // Dividimos el dataset en train i test
        Dataset<Row>[] splits = assembled.randomSplit(new double[]{0.3, 0.7});
        Dataset<Row> train = splits[1];
        Dataset<Row> test = splits[0];
        train.persist();
        train.show();

        // Ajustamos el modelo (SVM con CV)
        CrossValidatorModel cvModel = fitModel(train);

        // Predicciones sobre test set
        Dataset<Row> predictions = cvModel.transform(test).select(PREDICTION_COL, VAR_DIAS_ESTANCIA_DISC);

        // Definimos un evaluador
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setMetricName("accuracy")
                .setLabelCol(VAR_DIAS_ESTANCIA_DISC)
                .setPredictionCol(PREDICTION_COL);

        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Train samples: " + train.count());
        System.out.println("Test samples: " + test.count());
        System.out.println("Test Error = " + (1 - accuracy));

        ss.stop();
    }
}
