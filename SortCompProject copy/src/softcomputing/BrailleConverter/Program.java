package softcomputing.BrailleConverter;

import static com.googlecode.javacv.cpp.opencv_core.IPL_DEPTH_8U;
import static com.googlecode.javacv.cpp.opencv_core.cvCreateImage;
import static com.googlecode.javacv.cpp.opencv_core.cvGetSize;
import static com.googlecode.javacv.cpp.opencv_highgui.cvLoadImage;
import static com.googlecode.javacv.cpp.opencv_highgui.cvSaveImage;
import static com.googlecode.javacv.cpp.opencv_imgproc.CV_RGB2GRAY;
import static com.googlecode.javacv.cpp.opencv_imgproc.CV_THRESH_BINARY;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvCvtColor;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvThreshold;
import static com.googlecode.javacv.cpp.opencv_imgproc.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;

import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

import com.googlecode.javacv.cpp.opencv_core.CvMat;
import com.googlecode.javacv.cpp.opencv_core.IplImage;

public class Program {
	
	// global variables
	public static String imageDirPath = "C:\\Users\\luke\\Documents\\GitHub\\Softcomputing-Project\\SortCompProject\\src\\alphabet";		// path to folder with base, full-size images
	public static String processedImageDirPath = "C:\\Users\\luke\\Documents\\GitHub\\Softcomputing-Project\\SortCompProject\\src\\processed";	// path to folder with processed images
	public static int processedImageHeight = 20;
	public static int processedImageWidth = 15;
	
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
			ArrayList<Integer> neuronsInLayers = new ArrayList<Integer>();							// neurons in particular layers
			neuronsInLayers.clear();
			neuronsInLayers.add(300);																// 300 = 15 * 20 input
			neuronsInLayers.add(15);																// neurons in hidden layer
			neuronsInLayers.add(6);																	// 6 outputs (6 dots in braille alphabet) 
		
		// Preprocessing configuration
			
			
		// Learning configuration
			double learningRate = 0.2;																// learning rate
			int maxIteration = 1000;																// maximal number of iterations	
			double maxError = 0.2;																	// maximal acceptable error (to break learning process)
			boolean batchMode = false;																// learning batch mode
			int dataSetInputSize = 300;																// input data vector size 	(should be same as neurons in input layer)
			int dataSetOutputSize = 6;																// 6 - output data vector size 	(should be same as neurons in output layer), we recognised 6 dots in letter
		
			ArrayList<String> lettersToProcess = new ArrayList<String>();
			lettersToProcess.clear();
			lettersToProcess.add("A");												// letters to process
			lettersToProcess.add("B");
			lettersToProcess.add("C");
			lettersToProcess.add("D");
			lettersToProcess.add("E");
			/*lettersToProcess.add("F");
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
			//lettersToProcess.add("Q");
			lettersToProcess.add("R");
			lettersToProcess.add("S");
			lettersToProcess.add("T");
			lettersToProcess.add("U");
			//lettersToProcess.add("V");
			lettersToProcess.add("W");
			lettersToProcess.add("X");
			lettersToProcess.add("Y");
			lettersToProcess.add("Z");*/
			
			int randomWeightFrom = -1;
			int randomWeightTo = 1;

			
			
			
			
			/* select the mode of work */
			if (args.length > 0 && args[0].equalsIgnoreCase("Preprocess")) {
				Preprocess(imageDirPath);
			} else if (args.length > 0 && args[0].equalsIgnoreCase("Learn")) {
				
				//learn network
				double error = NeuralNetworkLearning(neuronsInLayers,
						TransferFunctionType.TANH, learningRate, maxIteration,
						maxError, batchMode, dataSetInputSize,
						dataSetOutputSize, lettersToProcess,
						randomWeightFrom, randomWeightTo);
				
				System.out.println("Error computed : " + error);

			} else {
				System.out.println("Wrong parameters");
			}

			System.out.println("Successfully finished");
		
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
			IplImage resizedImg = IplImage.create(15,20, grayImage.depth(), grayImage.nChannels());
			cvResize(grayImage, resizedImg, CV_INTER_LINEAR);
			
			/* Threshold it */
			IplImage imgThreshold = cvCreateImage(cvGetSize(resizedImg), resizedImg.depth(), resizedImg.nChannels());
			cvThreshold(resizedImg, imgThreshold, thresholdForPreprocessing, 255, CV_THRESH_BINARY);
			
			
			/* Save image into file */
			cvSaveImage(processedImageDirPath + "\\" + files.get(i).get_fileName() + ".png", imgThreshold);
			System.out.println("Successfully processed image :"
					+ files.get(i).get_fileName());
		}
	}

	public static double NeuralNetworkLearning(
			ArrayList<Integer> neuronsInLayers,
			TransferFunctionType transferFuncType, double learningRate,
			int maxIteration, double maxError, boolean batchMode,
			int dataSetInputSize, int dataSetOutputSize,
			ArrayList<String> lettersToProcess, int randomWeightFrom,
			int randomWeightTo) {
		System.out.println("Setting learning options...");
		MultiLayerPerceptron mlpNet = null;

		// activation function
		mlpNet = new MultiLayerPerceptron(neuronsInLayers, transferFuncType);

		// Backpropagation configuration
		BackPropagation BPLearningMethod = new BackPropagation();

		// Select training dataset
		DataSet trainingDataset = new DataSet(dataSetInputSize,
				dataSetOutputSize);
		for (int i = 0; i < lettersToProcess.size(); i++) 
		{
			String processedLetter = lettersToProcess.get(i);
			ArrayList<ImageFileInfo> fileNames = GetFileNamesList(Consts.TrainingFolderPath.get(processedLetter));
			for (int j = 0; j < fileNames.size(); j++) {
				IplImage img = cvLoadImage(fileNames.get(j).get_filePath());
				DataSetRow row = GenerateDataSetRowFromImage(img, lettersToProcess.get(i),
						dataSetInputSize);
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
		System.out.println("MLP learnt...");
		
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
			fileInfo.set_fileName(f.getName().substring(0, f.getName().length()-4));
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
		for(int i=0; i<imageMat.rows(); i++)
		{
			for(int j=0; j< imageMat.cols(); j++)
			{
				double value8Bit = imageMat.get(i, j);
				
				if(value8Bit == 255)
				{
					input[counter] = 1;
				}
				else
				{
					input[counter] = 0;
				}
				counter++;
			}
		}
		double[] desiredOutput = Consts.BrailleDots.get(processedLetter);

		DataSetRow row = new DataSetRow(input, desiredOutput);
		return row;
	}
}
