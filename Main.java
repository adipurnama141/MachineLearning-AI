public class Main {

	public static void main(String args[]){
		WekaAPI wa = new WekaAPI();

		try {
			wa.loadDataSet("iris.arff");
			wa.discretize();
			wa.numericToNominal();
			wa.createModel();
			wa.saveModel();
			wa.loadModel();
			wa.fullTraining();
			wa.tenFoldCV();
			wa.classifyFromKeyboard();
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		



	}

}