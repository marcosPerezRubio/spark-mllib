

import org.apache.spark.sql.SparkSession;
import exercise_1.exercise_1;

public class UPCSchool_MLlib {
	public static void main(String[] args) throws Exception {
		SparkSession spark = SparkSession.builder().master("local[*]")
				.appName("pacientesSim")
				.getOrCreate();
		exercise_1.pacientesSim(spark);
		spark.close(); 
	}

}
