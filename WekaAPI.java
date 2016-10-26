import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Random;
import java.util.ArrayList;
import java.util.Scanner;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class WekaAPI {
	private Instances dataset;
	private J48 tree;

	public void loadDataSet(String filename) throws Exception {
		BufferedReader reader = new BufferedReader(new FileReader(filename));
		dataset = new Instances(reader);
		dataset.setClassIndex(dataset.numAttributes() - 1);
		reader.close();
		System.out.println(dataset.toString());
	}

	public void discretize() throws Exception {
		Discretize filter = new Discretize();
		String[] options = new String[4];
		options[0] = "-R";
		options[1] = "first-last";
		options[2] = "-precision";
		options[3] = "6";
		filter.setOptions(options);
		filter.setInputFormat(dataset);
		dataset = Filter.useFilter(dataset , filter);

		System.out.println(dataset.toString());
	}

	public void numericToNominal() throws Exception {
		NumericToNominal filter = new NumericToNominal();
		String[] options = new String[2];
		options[0] = "-R";
		options[1] = "first-last";
		filter.setOptions(options);
		filter.setInputFormat(dataset);
		dataset = Filter.useFilter(dataset , filter);
		System.out.println(dataset.toString());
	}

	public void createModel() throws Exception {
		String[] options = new String[4];
		options[0] = "-C";
		options[1] = "0.25";
		options[2] = "-M";
		options[3] = "2";
		tree = new J48();
		tree.setOptions(options);
		tree.buildClassifier(dataset);
		
	}

	public void saveModel() throws Exception {
		weka.core.SerializationHelper.write("j48.model", tree);
	}

	public void loadModel() throws Exception {
		tree = (J48) weka.core.SerializationHelper.read("j48.model");
	}

	public void fullTraining() throws Exception {
		Evaluation eval = new Evaluation(dataset);
		eval.evaluateModel(tree,dataset);
		System.out.println(eval.toSummaryString("\nFull Training Results\n", false));
	}

	public void tenFoldCV() throws Exception {
		Evaluation eval = new Evaluation(dataset);
		eval.crossValidateModel(tree, dataset, 10, new Random(1));
		System.out.println(eval.toSummaryString("\n10 - Folds CrossEvaluation Results\n", false));
	}

	public void singleClassify(double sl , double sw , double pl , double pw) throws Exception {
		Instance instance = new DenseInstance(5);
		
		Attribute sepallength = new Attribute("sepallength");
	    Attribute sepalwidth = new Attribute("sepalwidth");
	    Attribute petallength = new Attribute("petallength");
	    Attribute petalwidth = new Attribute("petalwidth");
	    

	    instance.setValue(dataset.attribute(0), sl);
        instance.setValue(dataset.attribute(1), sw);
        instance.setValue(dataset.attribute(2), pw);
        instance.setValue(dataset.attribute(3), pl);
        instance.setDataset(dataset);

        double predicted = tree.classifyInstance(instance);
        System.out.println( dataset.classAttribute().value( (int) predicted ) );
	}	

	public void classifyFromKeyboard() throws Exception {
		Scanner scan = new Scanner(System.in);
		double sl,sw,pl,pw;

		System.out.print("Sepal Length : ");
		sl = scan.nextDouble();
		System.out.print("Sepal Width : ");
		sw = scan.nextDouble();
		System.out.print("Petal Length : ");
		pl = scan.nextDouble();
		System.out.print("Petal Width : ");
		pw = scan.nextDouble();
		singleClassify(sl,sw,pl,pw);
	}

}