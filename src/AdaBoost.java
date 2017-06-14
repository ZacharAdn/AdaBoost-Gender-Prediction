import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

/**
 * Created by Zahar Adiniaev, Marcel Binyaminov
 */
public class AdaBoost {


    /**
     * Question 1.a
     * Find separator with minimal mistake
     *
     * @param data
     * @return {a,b}
     */
    public double[] BruteForce(double[][] data){
        return BruteForceGeneric(1 , data);
    }

    /**
     * Question 1.b
     * Find separator with minimal weighted mistake
     *
     * @param data
     * @return {a,b, weight, lineTag}
     */
    public double[]  BruteForce2(double[][] data){
        return BruteForceGeneric(2, data);
    }

    /**
     * Generic function for BruteForce
     *
     * @param type
     * @param data
     * @return
     */
    double[] BruteForceGeneric(int type, double[][] data){
        double[] w = {0,0};
        int minError = Integer.MAX_VALUE;
        double minWeight = Double.MAX_VALUE;
        int tagMinWeight = 0;

        data = shuffle(data);

        for (int i = 0; i < data.length; i++) {
            for (int j = i+1; j < data.length; j++) {
                if(data[i][0] == data[j][0]) { //  To prevent distribution at 0
                    continue;
                }

                double[] tmpW = generateLine(data[i] , data[j]);

                int error = 0, tagAboveLine = 0 ;
                double weight = 0;

                if(type == 1){// bruteForce 1
                    error = errorCheck(data,tmpW,tagCheck(data,tmpW));
                    if(error < minError){
                        minError = error;
                        w = tmpW;
                    }
                }else{// bruteForce 2
                    double error1 = weightedErrorCheck(tmpW,data,1) , error2 = weightedErrorCheck(tmpW,data,-1);

                    if(error1 < error2){
                        weight = error1;
                        tagAboveLine = 1;
                    }else{
                        weight = error2;
                        tagAboveLine = -1;
                    }

                    if(weight < minWeight){
                        w = tmpW;
                        minWeight = weight;
                        tagMinWeight = tagAboveLine;
                    }
                }
            }
        }

        if(type == 1) {
            System.out.println("Errors: " + minError);
            return w;
        }else{
            return new double[] {w[0],w[1],minWeight,tagMinWeight};
        }
    }

    /**
     * Question 2
     * AdaBoost algorithm
     * Generate 5 lines (week classifiers) with weight for each
     * Print the error of the final solution (the strong classifier)
     *
     * @param data
     * @return
     */
    public double[][] adaBoost(double[][] data){
        double[][] weightedData = addWeights(data,(1/(double)data.length));// add weight 1/n to every point
        double[][] lines; // line => {a, b, lineWeight, lineTag}

        lines = weekClassifiersCalc(5 , weightedData);
        int totalError = adaBoostError(weightedData , lines);
        System.out.println("AdaBoost total error: "+totalError);

        return lines;
    }

    /**
     * Calculates the weak Classifiers, their weights and tag
     *
     * @param num
     * @param data
     * @return
     */
    private double[][] weekClassifiersCalc(int num, double[][] data) {
        double[][] ans  = new double[5][4];
        double[] ansTmp;
        double[] h_t = new double[2]; //w
        double e_t = 0 ;//line minimal weighted error
        double a_t = 0; //line weight

        for (int i = 0; i < 5; i++) { // generate lines
            ansTmp = BruteForce2(data);

            h_t[0] = ansTmp[0];
            h_t[1] = ansTmp[1];
            e_t = ansTmp[2];
            a_t = 0.5 * Math.log((1 -  e_t) / e_t);

            data = updateNewWeights(data, h_t, a_t);

            ans[i][0] = h_t[0];
            ans[i][1] = h_t[1];
            ans[i][2] = a_t;
            ans[i][3] = (tagCheck(data,h_t) == 1) ? 1 : -1;
        }
        return ans;
    }

    /**
     * Calculates the error of the adaBoost classifier
     *
     * @param data
     * @param lines
     * @return
     */
    private int adaBoostError(double[][] data, double[][] lines) {
        int ans = 0;

        for (double[] xi : data){
            double x = xi[0];
            double y = xi[2];
            double tag = (xi[1] == 1)? 1 : -1;

            double F_x = 0;
            for (double[] line : lines){
                double a = line[0];
                double b = line[1];
                double weight = line[2];
                double lineTag = line[3];

                double rule =  y - (a * x);
                int prediction = (rule > b) ? 1 : -1;
                prediction *= (int)lineTag;

                F_x += (prediction * weight);
            }

            if((F_x > 0 && tag == -1) || (F_x < 0 && tag == 1)){
                ans++;
            }
        }
        return ans;
    }

    /**
     * Add weight column for each point
     *
     * @param data
     * @param weight
     * @return
     */
    public double[][] addWeights(double[][] data, double weight) {
        double[][] ans = new double[data.length][4];

        for (int i = 0; i < data.length; i++) {
            ans[i] = new double[] { data[i][0], data[i][1], data[i][2], weight};
        }

        return ans;
    }

    /**
     * Update and normalize the new weights regarding the error of the prediction
     *
     * @param data
     * @param w
     * @param lineWeight
     * @return
     */
    private double[][] updateNewWeights(double[][] data, double[] w, double lineWeight) {
        double totalWeights = 0;
        double[][] ans = new double[data.length][];
        int k = 0;
        double a = w[0];
        double b = w[1];

        for (double[] xi : data){
            double x = xi[0];
            double y = xi[2];

            double rule = y - (a * x);
            int tag = tagCheck(data,w);

            if((rule > b && (tag != xi[1])) || (rule < b && (tag == xi[1]))){
                xi[3] *= Math.exp(lineWeight);
                totalWeights += xi[3];
            }else{
                xi[3] *= Math.exp(-lineWeight);
                totalWeights += xi[3];
            }
            ans[k++] = xi;
        }

        //normalization of the weights
        for (int i = 0; i < ans.length; i++) {
            ans[i][3] /= totalWeights;
        }

        return ans;
    }

    /**
     * Calculate the error of the data on the line w
     * Regarding the line tag
     *
     * @param w
     * @param data
     * @param lineTag
     * @return
     */
    private static double weightedErrorCheck(double[] w, double[][] data, int lineTag) {
        double error = 0;
        for (double[] xi : data) {
            double rule = xi[2] - (w[0] * xi[0]);
            double b = w[1];

            int tag = xi[1]==1 ? 1 : -1;
            int predict = (rule > b) ? 1 : -1;

            predict = predict * lineTag;
            if (predict != tag) {
                error += xi[3];
            }
        }
        return error;
    }

    /**
     * Checks the tag of the line
     *
     * @param data
     * @param w
     * @return
     */
    private int tagCheck(double[][] data, double[] w) {

        int countMale = 0, countFemale = 0;
        for (double[] xi : data) {
            double yi = xi[1];
            double rule = xi[2] - (w[0] * xi[0]);
            double b = w[1];

            if (rule > b) {
                if (yi == 1) {//men
                    countMale++;
                } else {
                    countFemale++;
                }
            } else if(rule < b){
                if (yi == 1) {
                    countMale--;
                } else {
                    countFemale--;
                }
            }
        }

        if(countMale > countFemale){
            return 1;
        }else{
            return  2;
        }
    }

    /**
     * Generate line between two points
     *
     * @param x1
     * @param x2
     * @return
     */
    private double[] generateLine(double[] x1, double[] x2) {
        double a = (x1[2] - x2[2]) / (x1[0] - x2[0]);
        double b = x1[2] - (a * x1[0]) ;

        return new double[] {a,b};
    }

    /**
     * Shuffle the data
     *
     * @param data
     * @return
     */
    private double[][] shuffle(double[][] data) {
        double[][] ans = new double[data.length][data[0].length];

        ArrayList<double[]> list  = new ArrayList();
        for (double[] xi : data) {
            list.add(xi);
        }

        for (int i = 0; i < data.length; i++) {
            int rand  = (int) (Math.random() * list.size());
            ans[i] = list.get(rand);
            list.remove(rand);
        }
        return ans;
    }

    /**
     * Calculates the num of errors of the prediction the on the data
     *
     * @param data
     * @param w
     * @param tagAboveLine
     * @return
     */
    private int errorCheck(double[][] data, double[] w, int tagAboveLine) {
        int error = 0;

        for (double[] xi : data) {
            double rule = xi[2] - (w[0] * xi[0]) ;
            double b = w[1];
            double tag = (xi[1]==1) ? 1 : -1;

            if(tagAboveLine == 1) {
                if (((rule > b) && (tag == -1)) || ((rule < b) && (tag == 1))) {
                    error++;
                }
            }else {
                if (((rule >= b) && (tag == 1)) || ((rule < b) && (tag == -1))) {
                    error++;
                }
            }
        }
        return error;
    }

    /**
     * File read
     *
     * @param path
     * @return
     */
    public double[][] readFile(String path){

        int size = 130; // 130 people, 65 male and 65 female
        double[][] data = new double[size][3];

        Scanner scan;
        File file = new File(path);
        try {
            scan = new Scanner(file);

            for(int i=0; i<size; i++) {
                for(int j=0; j<3; j++) {
                    data[i][j] = scan.nextDouble();
                }
            }

        } catch (FileNotFoundException e1) {
            e1.printStackTrace();
        }
        return data;
    }

}

class main{
    public static void main(String[] args) {
        AdaBoost machine = new AdaBoost();
        double[][] data = machine.readFile("HC_Body_Temperature.txt");

        //Question 1
        System.out.println("\nBruteForce 1:");
        machine.BruteForce(data);

        System.out.println("\nBruteForce 2:");
        double[][] weightedData = machine.addWeights(data, 1);
        double ans[] = machine.BruteForce2(weightedData);
        System.out.println("Errors: " + ans[2]);

        //Question 2
        System.out.println("\nAdaBoost:");
        double[][] lines = machine.adaBoost(data);
        System.out.println("\nlines:");
        for (int i = 0; i < lines.length; i++) {
            System.out.println("w = {" +lines[i][0] + ", " + lines[i][1] + "}, weight:" + lines[i][2] +
                    ", lineTag: "+ lines[i][3]);
        }
    }
}
