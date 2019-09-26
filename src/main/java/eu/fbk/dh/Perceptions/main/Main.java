package eu.fbk.dh.Perceptions.main;

import eu.fbk.dh.Perceptions.crawler.TwitterCrawler;
import eu.fbk.dh.Perceptions.database.JDBCConnectionManager;
import eu.fbk.dh.Perceptions.datasets.ArabicDataset;

import java.sql.Connection;
import java.sql.SQLException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * @author Mohamad Baalbaki
 */


public class Main {
    public static void main(String[] args) throws SQLException {

        Connection con = JDBCConnectionManager.getConnection();

        ////////////////////////////////CRAWLING///////////////////////////////////////////////// EVERY 7 days
        /*int cores = Runtime.getRuntime().availableProcessors(); //get cores for multithreading
        if (cores > 4) {
            cores = cores / 2;
        }

        for (int i = 0; i < args.length; i++) {
            try {
                String keyword = args[i]; //get the keyword from the command line arguments
                TwitterCrawler crawler = new TwitterCrawler(keyword, "", null); //run the crawler
                ExecutorService executor = Executors.newFixedThreadPool(cores); //execute it with the cores
                executor.execute(crawler);
                executor.shutdown();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }*/
        ////////////////////////////////CRAWLING/////////////////////////////////////////////////

        //Todo: add tweets to a dataset in an excel file (discard repetitions) and manually classify them

        ArabicDataset arabicDataset=new ArabicDataset(); //create the arabic dataset object
        arabicDataset.addUnseenTweetsInformationFromDatabase(); //load the tweets and their information to the dataset from the database



        con.close();
    }
}
