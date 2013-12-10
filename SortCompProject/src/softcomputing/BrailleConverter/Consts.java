package softcomputing.BrailleConverter;

import java.util.HashMap;
import java.util.Map;

public class Consts {
	public static Map<String,double[]> BrailleDots;
	public static Map<String,String> TrainingFolderPath;
	// ---- C:\\Users\\Marcel Starczyk\\Desktop\\SoftcomputingProject\\
	// ---- C:\\Users\\luke\\Git\\Softcomputing-Project\\SortCompProject\\
	
	public static String HomePath = "C:\\Users\\luke\\Git\\Softcomputing-Project\\SortCompProject\\";
	public static String BaseTestFolderPath = HomePath + "Test\\";
    public static void Initialize()
	{
		/*
		 * Braille language - we decode Braille sign in following way:
		 * 
		 * 'dots'
		 *    1  2
		 *    3  4
		 *    5  6
		 *    
		 * 'dots in vector'
		 *  [1 2 3 4 5 6]
		 *  
		 * 
		 */
		
		BrailleDots = new HashMap<String, double[]>();
		BrailleDots.put("A", new double[]{1,0,0,0,0,0});
		BrailleDots.put("B", new double[]{1,0,1,0,0,0});
		BrailleDots.put("C", new double[]{1,1,0,0,0,0});
		BrailleDots.put("D", new double[]{1,1,0,1,0,0});
		BrailleDots.put("E", new double[]{1,0,0,1,0,0});
		BrailleDots.put("F", new double[]{1,1,1,0,0,0});
		BrailleDots.put("G", new double[]{1,1,1,1,0,0});
		BrailleDots.put("H", new double[]{1,0,1,1,0,0});
		BrailleDots.put("I", new double[]{0,1,1,0,0,0});
		BrailleDots.put("J", new double[]{0,1,1,1,0,0});
		BrailleDots.put("K", new double[]{1,0,0,0,1,0});
		BrailleDots.put("L", new double[]{1,0,1,0,1,0});
		BrailleDots.put("M", new double[]{1,1,0,0,1,0});
		BrailleDots.put("N", new double[]{1,1,0,1,1,0});
		BrailleDots.put("O", new double[]{1,0,0,1,1,0});
		BrailleDots.put("P", new double[]{1,1,1,0,1,0});
		BrailleDots.put("Q", new double[]{1,1,1,1,1,0});
		BrailleDots.put("R", new double[]{1,0,1,1,1,0});
		BrailleDots.put("S", new double[]{0,1,1,0,1,0});
		BrailleDots.put("T", new double[]{0,1,1,1,1,0});
		BrailleDots.put("U", new double[]{1,0,0,0,1,1});
		BrailleDots.put("V", new double[]{1,0,1,0,1,1});
		BrailleDots.put("W", new double[]{0,1,1,1,0,1});
		BrailleDots.put("X", new double[]{1,1,0,0,1,1});
		BrailleDots.put("Y", new double[]{1,1,0,1,1,1});
		BrailleDots.put("Z", new double[]{1,0,0,1,1,1});
		
		
		TrainingFolderPath = new HashMap<String, String>();
		TrainingFolderPath.put("A", BaseTestFolderPath + "A");
		TrainingFolderPath.put("B", BaseTestFolderPath + "B");
		TrainingFolderPath.put("C", BaseTestFolderPath + "C");
		TrainingFolderPath.put("D", BaseTestFolderPath + "D");
		TrainingFolderPath.put("E", BaseTestFolderPath + "E");
		TrainingFolderPath.put("F", BaseTestFolderPath + "F");
		TrainingFolderPath.put("G", BaseTestFolderPath + "G");
		TrainingFolderPath.put("H", BaseTestFolderPath + "H");
		TrainingFolderPath.put("I", BaseTestFolderPath + "I");
		TrainingFolderPath.put("J", BaseTestFolderPath + "J");
		TrainingFolderPath.put("K", BaseTestFolderPath + "K");
		TrainingFolderPath.put("L", BaseTestFolderPath + "L");
		TrainingFolderPath.put("M", BaseTestFolderPath + "M");
		TrainingFolderPath.put("N", BaseTestFolderPath + "N");
		TrainingFolderPath.put("O", BaseTestFolderPath + "O");
		TrainingFolderPath.put("P", BaseTestFolderPath + "P");
		TrainingFolderPath.put("Q", BaseTestFolderPath + "Q");
		TrainingFolderPath.put("R", BaseTestFolderPath + "R");
		TrainingFolderPath.put("S", BaseTestFolderPath + "S");
		TrainingFolderPath.put("T", BaseTestFolderPath + "T");
		TrainingFolderPath.put("U", BaseTestFolderPath + "U");
		TrainingFolderPath.put("V", BaseTestFolderPath + "V");
		TrainingFolderPath.put("W", BaseTestFolderPath + "W");
		TrainingFolderPath.put("X", BaseTestFolderPath + "X");
		TrainingFolderPath.put("Y", BaseTestFolderPath + "Y");
		TrainingFolderPath.put("Z", BaseTestFolderPath + "Z");
		
	}

}
