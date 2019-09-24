package eu.fbk.dh.Perceptions.main;

import eu.fbk.dh.Perceptions.crawler.TwitterCrawler;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * @author Mohamad Baalbaki
 */


public class Main {
    public static void main(String[] args) {

        int cores = Runtime.getRuntime().availableProcessors(); //get cores for multithreading
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
        }
    }
}
