package eu.fbk.dh.Perceptions.crawler;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.stream.JsonWriter;
import eu.fbk.dh.Perceptions.database.JDBCConnectionManager;
import twitter4j.*;
import twitter4j.auth.OAuth2Token;
import twitter4j.conf.ConfigurationBuilder;

import java.io.*;
import java.nio.charset.Charset;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Properties;

/**
 * Created by giovannimoretti on 28/11/16.
 */
public class TwitterCrawler implements Runnable {


    private String tag = "";
    private String additional_query_parameters = "";
    private Long sinceId = null;
    private String timestamp;

    public TwitterCrawler(String hashtag, String additional_query_parameters, Long since_id) {
        this.tag = hashtag;
        this.additional_query_parameters = additional_query_parameters;
        this.sinceId = since_id;
    }

    @Override
    public void run() {



        Properties prop = new Properties();
        InputStream input = null;

        String consumerKey = "";
        String consumerSecretKey = "";


        try {

            input = getClass().getClassLoader().getResourceAsStream("config.properties");

            // load a properties file
            prop.load(input);

            consumerKey = prop.getProperty("consumerKey");
            consumerSecretKey = prop.getProperty("consumerSecretKey");

        }catch (Exception e){
            e.printStackTrace();
        }


        ConfigurationBuilder cb = new ConfigurationBuilder();
        cb.setDebugEnabled(true)
                .setOAuthConsumerKey(consumerKey)
                .setOAuthConsumerSecret(consumerSecretKey)
                .setApplicationOnlyAuthEnabled(true)
                .setJSONStoreEnabled(true);

        ConfigurationBuilder cb2 = new ConfigurationBuilder();
        cb2.setDebugEnabled(true)
                .setOAuthConsumerKey(consumerKey)
                .setOAuthConsumerSecret(consumerSecretKey)
                .setApplicationOnlyAuthEnabled(true);

        OAuth2Token token = null;


        try {
            token = new TwitterFactory(cb2.build()).getInstance().getOAuth2Token();
            cb.setOAuth2TokenType(token.getTokenType());
            cb.setOAuth2AccessToken(token.getAccessToken());
        } catch (Exception e) {
            e.printStackTrace();
        }


        TwitterFactory tf = new TwitterFactory(cb.build());
        Twitter twitter = tf.getInstance();


        try {
            Gson gson = new Gson();
            JsonParser parser = new JsonParser();
            Calendar rightNow = Calendar.getInstance();

            timestamp =rightNow.get(Calendar.YEAR) + "_" +(rightNow.get(Calendar.MONTH)+1)+"_"+rightNow.get(Calendar.DAY_OF_MONTH)+"-"+rightNow.get(Calendar.HOUR_OF_DAY)+"_"+rightNow.get(Calendar.MINUTE);
//          OutputStreamWriter os = new OutputStreamWriter(new FileOutputStream("ArabicJsonFiles/"+timestamp+"-"+this.tag + (this.sinceId != null ? "_" + sinceId : "") + ".json"), Charset.forName("UTF-8").newEncoder());
            StringWriter os = new StringWriter();

            JsonWriter writer = new JsonWriter(os); //this is the writer that will produce the output stream (tweets as strings) needed to store in the db
            writer.setIndent(" ");
            writer.beginArray();


            long min_id = Long.MAX_VALUE;

            boolean perform = true;
            int tweet_count = 0;

            while (perform) {


                Query query = new Query(this.tag + additional_query_parameters); //+ " -filter:retweets"
                //query.lang("it");
                query.resultType(Query.RECENT);
                query.count(100);

                if (min_id != Long.MAX_VALUE) {
                    query.setMaxId(min_id);
                }

                if (this.sinceId != null) {
                    query.setSinceId(sinceId);
                }

                try {

                    QueryResult result = twitter.search(query);

                    if (result.getTweets().size() <= 1) {
                        System.out.println("No more " + this.tag + " tweet");
                        perform = false;
                    }

                    JsonObject jo=new JsonObject();

                    for (Status status : result.getTweets()) {
                        jo = parser.parse(TwitterObjectFactory.getRawJSON(status)).getAsJsonObject();
                        min_id = Math.min(min_id, status.getId());
                        gson.toJson(jo, JsonObject.class, writer);
                        tweet_count++;
                    }

                    System.out.println(tag + ": " + result.getRateLimitStatus().getRemaining() + "/" + result.getRateLimitStatus().getLimit() + "  total_tweets:" + tweet_count);
                    if (result.getRateLimitStatus().getRemaining() <= 10) {
                        System.out.println("...Ufff...Cool Down tag " + this.tag + "...." + "Wait for :" + result.getRateLimitStatus().getSecondsUntilReset() + "s");
                        Thread.sleep((result.getRateLimitStatus().getSecondsUntilReset() * 1000) + 5000);
                    }

                } catch (TwitterException e) {
                    if (e.getErrorCode() == 503) {
                        System.out.println("Server overloaded... I wait for 5 min...sob...");
                        Thread.sleep(300000);
                    } else {
                        System.out.println("...Ufff...Cool Down tag " + this.tag + "....");
                        Thread.sleep(910000);
                    }
                } catch (NullPointerException n){
                    System.out.println("mmm null pointer ... so bad.... now I wait a little bit...");
                    Thread.sleep(300000);
                }


            }


            writer.endArray();
            writer.close();

            storeDateOfCrawlAndTweetsInDb(getFilePath().replace(".json",""),os.toString(),tweet_count); //store tweets in db

            BufferedWriter fileWriter = new BufferedWriter(new FileWriter("ArabicJsonFiles/"+timestamp+"-"+this.tag + (this.sinceId != null ? "_" + sinceId : "") + ".json"));
            fileWriter.write(os.toString()); //this is the write responsible of storing the output stream (tweets) in files for extra safety

            fileWriter.close();

        } catch (Exception e) {
            e.printStackTrace();
        }


    }

    public String getFilePath(){
        return timestamp+"-"+this.tag + (this.sinceId != null ? "_" + sinceId : "") + ".json";
    }

    public void storeDateOfCrawlAndTweetsInDb(String dateOfCrawl, String tweets, int tweetCount) throws SQLException {
            Connection con = JDBCConnectionManager.getConnection();

            PreparedStatement pstmt = con.prepareStatement("REPLACE INTO ar_tweets(dateOfCrawl,allTweets,nbOfTweets) VALUES(?,?,?)");
            pstmt.setString(1, dateOfCrawl);
            pstmt.setString(2, tweets);
            pstmt.setInt(3, tweetCount);

            pstmt.execute();
            pstmt.close();

            con.close();
    }
}