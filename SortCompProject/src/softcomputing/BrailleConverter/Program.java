package softcomputing.BrailleConverter;

import static com.googlecode.javacv.cpp.opencv_core.IPL_DEPTH_8U;
import static com.googlecode.javacv.cpp.opencv_core.cvCreateImage;
import static com.googlecode.javacv.cpp.opencv_core.cvGetSize;
import static com.googlecode.javacv.cpp.opencv_highgui.cvLoadImage;
import static com.googlecode.javacv.cpp.opencv_highgui.cvSaveImage;
import static com.googlecode.javacv.cpp.opencv_imgproc.*;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.Random;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

import com.googlecode.javacv.cpp.opencv_core.CvMat;
import com.googlecode.javacv.cpp.opencv_core.IplImage;

public class Program {

	// ----- C:\\SoftcomputingProject\\SortCompProject\\
	// ----- C:\\Users\\luke\\Git\\Softcomputing-Project\\SortCompProject\\

	public static String HomePath = "C:\\SoftcomputingProject\\SortCompProject\\";
	// global variables
	public static String imageDirPath = HomePath + "src\\alphabet"; // path to
																	// folder
																	// with
																	// base,
																	// full-size
																	// images
	public static String processedImageDirPath = HomePath + "src\\processed"; // path
																				// to
																				// folder
																				// with
																				// processed
																				// images

	public static String whiteNoiseDirPath = HomePath + "src\\whiteNoise";

	public static String csvFilePath = HomePath + "results.csv";
	public static String networkFilePath = HomePath
			+ "NeuralNetworks\\network_44_set3.nnet";
	public static String networkName = "network_44_set3";
	public static int processedImageHeight = 20; // number of rows
	public static int processedImageWidth = 15; // number of columns

	public static double[] networkOutput;
	public static float distortionRate = 0.0f; // rate to 
	public static Random random = new Random();
	

	// public static DateFormat dateFormat = new
	// SimpleDateFormat("yyyy/MM/dd HH:mm:ss");

	public static Date date;

	// Configuration
	final static int HIDDEN_LAYER_NEURONS_MAX = 20;
	final static int HIDDEN_LAYER_NEURONS_MIN = 10;

	final static double LEARNING_RATE_MAX = 0.20;
	final static double LEARNING_RATE_MIN = 0.05;
	final static int MAX_ITERATION_MAX = 20000;

	final static int INPUT_NEURONS = 300; // 300 = 15 * 20 input
	final static int OUTPUT_NEURONS = 6; // 6 outputs 6 dots in braille
											// alphabet)

	public static int MAX_ITERATION = 20000; // maximal number of iterations
	public static double MAX_ERROR = 0.1; // maximal acceptable error (to break
											// learning process)
	public static boolean BATCH_MODE = false; // learning batch mode

	final static int DATASET_INPUT_SIZE = 300; // input data vector size(should
												// be same as neurons in input
												// layer)
	final static int DATASET_OUTPUT_SIZE = 6; // 6 - output data vector size
												// (should be same as neurons in
												// output layer), we recognised
												// 6 dots in letter

	public static int thresholdForPreprocessing = 240;

	/**
	 * @param args
	 * @throws FileNotFoundException
	 */
	public static void main(String[] args) throws FileNotFoundException {

		System.out.println("Welcome at MLP program");

		Consts.Initialize();
		// set up variables

		// Network configuration
		ArrayList<Integer> neuronsInLayers = new ArrayList<Integer>(); // neurons
																		// in
																		// particular
																		// layers
		// Learning configuration

		ArrayList<String> lettersToProcess = new ArrayList<String>();
		lettersToProcess.clear();
		lettersToProcess.add("A"); // letters to process
		lettersToProcess.add("B");
		lettersToProcess.add("C");
		lettersToProcess.add("D");
		lettersToProcess.add("E");
		lettersToProcess.add("F");
		lettersToProcess.add("G");
		lettersToProcess.add("H");
		lettersToProcess.add("I");
		lettersToProcess.add("J");
		lettersToProcess.add("K");
		lettersToProcess.add("L");
		lettersToProcess.add("M");
		lettersToProcess.add("N");
		lettersToProcess.add("O");
		lettersToProcess.add("P");
		// lettersToProcess.add("Q");
		lettersToProcess.add("R");
		lettersToProcess.add("S");
		lettersToProcess.add("T");
		lettersToProcess.add("U");
		// lettersToProcess.add("V");
		lettersToProcess.add("W");
		lettersToProcess.add("X");
		lettersToProcess.add("Y");
		lettersToProcess.add("Z");

		int randomWeightFrom = -1;
		int randomWeightTo = 1;

		/* select the mode of work */
		if (args.length > 0 && args[0].equalsIgnoreCase("Preprocess")) {
			Preprocess(imageDirPath);
		} else if (args.length > 0 && args[0].equalsIgnoreCase("Learn")) {

			int currentIteration = 0;

			for (double i = LEARNING_RATE_MIN; i <= LEARNING_RATE_MAX; i = i + 0.05) {
				for (int j = HIDDEN_LAYER_NEURONS_MIN; j <= HIDDEN_LAYER_NEURONS_MAX; j++) {
					for (int k = MAX_ITERATION_MAX; k > 0; k -= 1000) {

						try {
							neuronsInLayers.clear();
							neuronsInLayers.add(INPUT_NEURONS);
							neuronsInLayers.add(j);
							neuronsInLayers.add(OUTPUT_NEURONS);

							// Getting date to add to network name
							date = new Date();
							double error = NeuralNetworkLearning(
									neuronsInLayers, TransferFunctionType.TANH,
									i, k, MAX_ERROR, BATCH_MODE,
									DATASET_INPUT_SIZE, DATASET_OUTPUT_SIZE,
									lettersToProcess, randomWeightFrom,
									randomWeightTo, currentIteration);

							saveResultsToCsvFile(error, currentIteration, i, j,
									k);
							// Changing configuration after every iteration
							// LEARNING_RATE += 0.1;
							// MAX_ITERATION -= 1000;
							currentIteration += 1;
						} catch (Exception e) {
							System.out.println(e.toString());
						}
					}
				}
			}

		} else if (args.length > 0 && args[0].equalsIgnoreCase("Recognize")) {
			System.out.println("Starting the recognizing process...");
			for (int i = 0; i < lettersToProcess.size(); i++) {
				
				String processedLetter = lettersToProcess.get(i);
				ArrayList<ImageFileInfo> fileNames = GetFileNamesList(Consts.TrainingFolderPath
						.get(processedLetter));
				for (int j = 0; j < fileNames.size(); j++) {
					
					if( distortionRate < 0.9){
						distortionRate = distortionRate + 0.1f;
					}
					else
						distortionRate = 0.0f;
					
					networkOutput = NeuralNetworkRecognizing(networkFilePath,
							DATASET_INPUT_SIZE, lettersToProcess, fileNames, i,
							j);
					saveRecognitionToFile(networkOutput, processedLetter, j,
							"WHITENOISE");
					
				}
				
			}

		} else {

			System.out.println("Wrong parameters");
		}

		System.out.println("===== Successfully finished =====");

	}

	public static void saveRecognitionToFile(double[] networkOutput,
			String processedLetter, int j, String networkName) {
		try {
			FileWriter writer = new FileWriter(HomePath + networkName
					+ "_results.csv", true);

			writer.append("==========================================================================================================================================\n");
			writer.append(processedLetter + j + "\n");
			writer.append("Expectation: ["
					+ Consts.BrailleDots.get(processedLetter)[0] + " , "
					+ Consts.BrailleDots.get(processedLetter)[1] + " , "
					+ Consts.BrailleDots.get(processedLetter)[2] + " , "
					+ Consts.BrailleDots.get(processedLetter)[3] + " , "
					+ Consts.BrailleDots.get(processedLetter)[4] + " , "
					+ Consts.BrailleDots.get(processedLetter)[5] + "]\n");
			writer.append("Prediction: = [ " + networkOutput[0] + " , "
					+ networkOutput[1] + " , " + networkOutput[2] + " , "
					+ networkOutput[3] + " , " + networkOutput[4] + " , "
					+ networkOutput[5] + " ]\n");
			writer.append("==========================================================================================================================================\n\n");

			writer.flush();
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void saveResultsToCsvFile(double error, int currentIteration,
			double learning_rate, int neuronsInHiddenLayer, int MaxIter) {
		try {
			FileWriter writer = new FileWriter(csvFilePath, true);

			// next row
			writer.append("\n");
			// ID
			writer.append(Integer.toString(currentIteration) + ", ");
			// NETWORK
			writer.append("network_" + currentIteration + ".nnet" + ", ");
			// INPUT_NEURONS
			writer.append(Integer.toString(INPUT_NEURONS) + ", ");
			// HIDDEN_LAYER_NEURONS
			writer.append(Integer.toString(neuronsInHiddenLayer) + ", ");
			// OUTPUT_NEURONS
			writer.append(Integer.toString(OUTPUT_NEURONS) + ", ");
			// LEARNING_RATE
			writer.append(Double.toString(learning_rate) + ", ");
			// MAX_ITERATION
			writer.append(Integer.toString(MAX_ITERATION) + ", ");
			// MAX_ERROR
			writer.append(Double.toString(MAX_ERROR) + ", ");
			// BATCH_MODE
			writer.append(Boolean.toString(BATCH_MODE) + ", ");
			// DATASET_INPUT_SIZE
			writer.append(Integer.toString(DATASET_INPUT_SIZE) + ", ");
			// DATASET_OUTPUT_SIZE
			writer.append(Integer.toString(DATASET_OUTPUT_SIZE) + ", ");
			// NUMBER_OF_EXECUTIONS
			writer.append(Integer.toString(MaxIter) + ", ");
			// NETWORK_ERROR
			writer.append(Double.toString(error));

			writer.flush();
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void Preprocess(String imageDirectoryPath) {
		System.out.println("Preprocessing started.");
		/* extract file path from folder */
		ArrayList<ImageFileInfo> files = GetFileNamesList(imageDirectoryPath);

		for (int i = 0; i < files.size(); i++) {
			/* Load image */
			IplImage originalImg = cvLoadImage(files.get(i).get_filePath());

			/* Convert into gray scale */
			IplImage grayImage = IplImage.create(originalImg.cvSize(),
					IPL_DEPTH_8U, 1);
			cvCvtColor(originalImg, grayImage, CV_RGB2GRAY);

			/* Resize */
			IplImage resizedImg = IplImage.create(15, 20, grayImage.depth(),
					grayImage.nChannels());
			cvResize(grayImage, resizedImg, CV_INTER_LINEAR);

			/* Threshold it */
			IplImage imgThreshold = cvCreateImage(cvGetSize(resizedImg),
					resizedImg.depth(), resizedImg.nChannels());
			cvThreshold(resizedImg, imgThreshold, thresholdForPreprocessing,
					255, CV_THRESH_BINARY);

			/* Save image into file */
			cvSaveImage(processedImageDirPath + "\\"
					+ files.get(i).get_fileName() + ".png", imgThreshold);
			System.out.println("Successfully processed image :"
					+ files.get(i).get_fileName());
		}
	}

	public static double[] whiteNoise(double[] inputVector, double distortionRatio)
			throws Exception {
		double[] outputVector = new double[inputVector.length];

		if (distortionRatio > 1 || distortionRatio < 0) {
			throw new Exception("Distortion ratio must be value from range <0,1>");
		}
		Random r = new Random();
		r.nextDouble();
		for (int i = 0; i < inputVector.length; i++) {
			if (r.nextDouble() < distortionRatio) {
				if (r.nextBoolean()) {
					outputVector[i] = 1;
				} else {
					outputVector[i] = 0;
				}
			} else {
				outputVector[i] = inputVector[i];
			}
		}
		return outputVector;
	}

	public static double[] noiseInOneRow(double[] inputVector, double distortionRatio)
			throws Exception {
		double[] outputVector = new double[inputVector.length];

		if (distortionRatio > 1 || distortionRatio < 0) {
			throw new Exception("Distortion ratio must be value from range <0,1>");
		}
		Random r = new Random();
		r.nextDouble();
		for (int i = 0; i < processedImageHeight; i++) {
			if (r.nextDouble() < distortionRatio) {
				for (int j = i * processedImageWidth; j < ((i + 1) * processedImageWidth); j++) {
					if (r.nextBoolean()) {
						outputVector[j] = 1;
					} else {
						outputVector[j] = 0;
					}
				}
			} else {
				for (int j = i * processedImageWidth; j < ((i + 1) * processedImageWidth); j++) {
					outputVector[j] = inputVector[j];
				}
			}
		}
		return outputVector;
	}

	public static double[] whiteLines(double[] inputVector, double distortionRatio) throws Exception {
		double[] outputVector = new double[inputVector.length];

		if (distortionRatio > 1 || distortionRatio < 0) {
			throw new Exception("Distortion ratio must be value from range <0,1>");
		}
		Random r = new Random();
		r.nextDouble();
		for (int i = 0; i < processedImageWidth; i++) {
			if (r.nextDouble() < distortionRatio) {
				for (int j = i * processedImageHeight; j < ((i + 1) * processedImageHeight); j++) {
						outputVector[j] = 1;	
				}
			} else {
				for (int j = i * processedImageHeight; j < ((i + 1) * processedImageHeight); j++) {
					outputVector[j] = inputVector[j];
				}
			}
		}
		return outputVector;
	}
    //TODO
	public static double[] letterCovering(double[] inputVector, double distortionRatio) throws Exception {
		double[] outputVector = new double[inputVector.length];

		if (distortionRatio > 1 || distortionRatio < 0) {
			throw new Exception("Distortion ratio must be value from range <0,1>");
		}
		Random r = new Random();
		r.nextDouble();
		for (int i = 0; i < processedImageHeight; i++) {
			if (i < distortionRatio*20) {
				for (int j = i * processedImageWidth; j < ((i + 1) * processedImageWidth); j++) {
						outputVector[j] = 0;
				}
			} else {
				for (int j = i * processedImageWidth; j < ((i + 1) * processedImageWidth); j++) {
					outputVector[j] = inputVector[j];
				}
			}
		}
		
		return outputVector;
	}

	public static double[] NeuralNetworkRecognizing(String networkPath,
			int dataSetInputSize, ArrayList<String> lettersToProcess,
			ArrayList<ImageFileInfo> fileNames, int i, int j) {

		double[] inputVector = new double[dataSetInputSize];

		InputStream networkInputStream = null;
		try {
			networkInputStream = new FileInputStream(networkPath);

		} catch (Exception e) {
			e.printStackTrace();
		}

		String testedLetter = lettersToProcess.get(i);
		IplImage img = cvLoadImage(fileNames.get(j).get_filePath());
		inputVector = GenerateInputVectorFromImage(img,
				lettersToProcess.get(j), dataSetInputSize);
		
		/*
		 * for (int i = 0; i < lettersToProcess.size(); i++) { String
		 * processedLetter = lettersToProcess.get(i); ArrayList<ImageFileInfo>
		 * fileNames =
		 * GetFileNamesList(Consts.TrainingFolderPath.get(processedLetter)); for
		 * (int j = 0; j < fileNames.size(); j++) { IplImage img =
		 * cvLoadImage(fileNames.get(j).get_filePath()); inputVector =
		 * GenerateInputVectorFromImage(img, lettersToProcess.get(i),
		 * dataSetInputSize);
		 * 
		 * } }
		 */

		// load network
		NeuralNetwork neuralNetwork = NeuralNetwork.load(networkInputStream);
		
		inputVector = IntroduceDistortion(inputVector, distortionRate, DistortionType.WHITE_NOISE);
		PrintOutVector(inputVector, testedLetter);
		// set input to the network
		neuralNetwork.setInput(inputVector);
		// calculate the network
		neuralNetwork.calculate();
		// get network output
		double[] networkOutput = neuralNetwork.getOutput();
		return networkOutput;

	}

	public static double NeuralNetworkLearning(
			ArrayList<Integer> neuronsInLayers,
			TransferFunctionType transferFuncType, double learningRate,
			int maxIteration, double maxError, boolean batchMode,
			int dataSetInputSize, int dataSetOutputSize,
			ArrayList<String> lettersToProcess, int randomWeightFrom,
			int randomWeightTo, int currentIteration) {

		System.out.println("Setting learning options...");
		MultiLayerPerceptron mlpNet = null;

		// activation function
		mlpNet = new MultiLayerPerceptron(neuronsInLayers, transferFuncType);

		// Backpropagation configuration
		BackPropagation BPLearningMethod = new BackPropagation();

		// Select training dataset
		DataSet trainingDataset = new DataSet(dataSetInputSize,
				dataSetOutputSize);
		for (int i = 0; i < lettersToProcess.size(); i++) {
			String processedLetter = lettersToProcess.get(i);
			ArrayList<ImageFileInfo> fileNames = GetFileNamesList(Consts.TrainingFolderPath
					.get(processedLetter));
			for (int j = 0; j < fileNames.size(); j++) {
				IplImage img = cvLoadImage(fileNames.get(j).get_filePath());
				DataSetRow row = GenerateDataSetRowFromImage(img,
						lettersToProcess.get(i), dataSetInputSize);
				trainingDataset.addRow(row);
			}
		}
		// assigning values
		BPLearningMethod.setLearningRate(learningRate);
		BPLearningMethod.setMaxIterations(maxIteration);
		BPLearningMethod.setMaxError(maxError);
		BPLearningMethod.setBatchMode(batchMode);
		BPLearningMethod.setTrainingSet(trainingDataset);

		System.out.println("Parameters for back propagation set..");
		// select initial weights
		mlpNet.randomizeWeights(randomWeightFrom, randomWeightTo);
		// learning method
		System.out.println("Weights randomized...");
		System.out.println("Learning started...");
		mlpNet.learn(trainingDataset, BPLearningMethod);

		System.out.println();
		System.out.println("MLP learnt...");
		String name = "network_" + currentIteration + ".nnet";
		mlpNet.save(name);

		/*
		 * Testing Network Error
		 */

		return BPLearningMethod.getTotalNetworkError();
	}

	public static ArrayList<ImageFileInfo> GetFileNamesList(String folder) {

		File directory = new File(folder);
		File[] contents = directory.listFiles();

		ArrayList<ImageFileInfo> filePaths = new ArrayList<ImageFileInfo>();
		for (File f : contents) {
			ImageFileInfo fileInfo = new ImageFileInfo();
			fileInfo.set_fileName(f.getName().substring(0,
					f.getName().length() - 4));
			fileInfo.set_directoryPath(f.getParent());
			fileInfo.set_filePath(f.getAbsolutePath());
			filePaths.add(fileInfo);
		}
		return filePaths;
	}

	public static DataSetRow GenerateDataSetRowFromImage(IplImage image,
			String processedLetter, int inputSize) {
		double[] input = new double[inputSize];

		CvMat imageMat = image.asCvMat();
		int counter = 0;
		for (int i = 0; i < imageMat.rows(); i++) {
			for (int j = 0; j < imageMat.cols(); j++) {
				double value8Bit = imageMat.get(i, j);

				if (value8Bit == 255) {
					input[counter] = 1;
				} else {
					input[counter] = 0;
				}
				counter++;
			}
		}
		double[] desiredOutput = Consts.BrailleDots.get(processedLetter);

		DataSetRow row = new DataSetRow(input, desiredOutput);
		return row;
	}

	public static double[] GenerateInputVectorFromImage(IplImage image,
			String processedLetter, int inputSize) {
		double[] input = new double[inputSize];

		CvMat imageMat = image.asCvMat();
		int counter = 0;
		for (int i = 0; i < imageMat.rows(); i++) {
			for (int j = 0; j < imageMat.cols(); j++) {
				double value8Bit = imageMat.get(i, j);

				if (value8Bit == 255) {
					input[counter] = 1;
				} else {
					input[counter] = 0;
				}
				counter++;
			}

		}
		return input;
	}

	public static void PrintOutVector(double[] vector, String name)
	{
		System.out.println(" ---- " + name + " ---- ");
		for (int i = 0; i < processedImageHeight; i++) {
				for (int j = i * processedImageWidth; j < ((i + 1) * processedImageWidth); j++) {
					if(vector[j] == 0)
					{
						System.out.print("1 ");
					}
					else
					{
						System.out.print("0 ");
					}
				}
				System.out.println("");
		}
	}
	
	public static double[] IntroduceDistortion(double[] inputVector, double distortionRatio, DistortionType distType)
	{
		double[] outputVector = new double[inputVector.length];
		try {
			switch(distType)
			{
			case NONE:
				outputVector = inputVector;
				break;
			case WHITE_NOISE:
				outputVector = whiteNoise(inputVector, distortionRatio);
				break;
			case NOISE_IN_ONE_ROW:
				outputVector = noiseInOneRow(inputVector, distortionRatio);
				break;
			case WHITE_LINES:
				outputVector = whiteLines(inputVector, distortionRatio);
				break;
			case LETTER_COVERING:
				outputVector = letterCovering(inputVector, distortionRatio);
				break;
			case TEST_ALL:
				PrintOutVector(inputVector,"input");
				double[] whNoi = whiteNoise(inputVector, 0.2);
				PrintOutVector(whNoi,"whNoi");
				double[] noiOneRow = noiseInOneRow(inputVector, 0.3);
				PrintOutVector(noiOneRow,"noiOneRow");
				double[] whLin = whiteLines(inputVector, 0.2);
				PrintOutVector(whLin,"whLin");
				break;
			default:
				break;
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return outputVector;
	}
}
