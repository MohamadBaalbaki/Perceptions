package eu.fbk.dh.Perceptions.main;

import eu.fbk.dh.Perceptions.crawler.TwitterCrawler;
import eu.fbk.dh.Perceptions.database.JDBCConnectionManager;
import eu.fbk.dh.Perceptions.datasets.ArabicDataset;
import org.apache.poi.openxml4j.exceptions.InvalidFormatException;

import java.io.IOException;
import java.sql.Connection;
import java.sql.SQLException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * @author Mohamad Baalbaki
 */


public class Main_Crawler {
    public static void main(String[] args) throws SQLException, IOException, InvalidFormatException {

        Connection con = JDBCConnectionManager.getConnection();

        ////////////////////////////////CRAWLING///////////////////////////////////////////////// EVERY 8 days
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

        con.close();
    }
}
