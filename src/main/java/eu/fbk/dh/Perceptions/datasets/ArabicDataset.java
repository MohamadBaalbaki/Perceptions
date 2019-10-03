package eu.fbk.dh.Perceptions.datasets;

import com.google.gson.*;
import eu.fbk.dh.Perceptions.database.JDBCConnectionManager;
import eu.fbk.dh.Perceptions.models.ExcelFile;
import eu.fbk.dh.Perceptions.models.RetweetedStatus;
import eu.fbk.dh.Perceptions.models.Tweet;
import eu.fbk.dh.Perceptions.models.User;
import org.apache.poi.openxml4j.exceptions.InvalidFormatException;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFSheet;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;

/**
 * @author Mohamad Baalbaki
 */


public class ArabicDataset extends ExcelFile {
    private ArrayList<String> tweetIdsStr; //twitter recommend using id_str instead of id
    private ArrayList<String> tweets;
    private ArrayList<String> locations;
    private ArrayList<Integer> inFavorOfMigration;

    public static ArrayList<Tweet> getAllTweets() throws SQLException {
        Connection con = JDBCConnectionManager.getConnection();
        ArrayList<Tweet> allTweets = new ArrayList<Tweet>();
        PreparedStatement preparedStatement = con.prepareStatement("SELECT allTweets from ar_tweets"); //get all the tweets from the database
        ResultSet resultSet = preparedStatement.executeQuery();
        JsonParser jsonParser = new JsonParser(); //to parse string into json array
        while (resultSet.next()) {
            JsonElement jsonElement = jsonParser.parse(resultSet.getString("allTweets"));
            JsonArray tweetsAsJsonArray = jsonElement.getAsJsonArray();
            for (int i = 0; i < tweetsAsJsonArray.size(); i++) {
                JsonObject tweetJsonObject = tweetsAsJsonArray.get(i).getAsJsonObject(); //got each tweet as a json object
                String tweetId = null, tweetString = null, tweetLocation = null;
                RetweetedStatus retweetedStatus = new RetweetedStatus();
                if (tweetJsonObject.has("retweeted_status")) { //check if it is a retweet
                    JsonObject retweetedStatusJsonObject = tweetJsonObject.get("retweeted_status").getAsJsonObject(); //get retweetedStatus json object
                    if (retweetedStatusJsonObject != null) { //if the tweet is a retweet
                        tweetId = retweetedStatusJsonObject.get("id_str").getAsString();
                        tweetString = retweetedStatusJsonObject.get("full_text").getAsString().trim();
                        tweetLocation = retweetedStatusJsonObject.get("user").getAsJsonObject().get("location").getAsString();
                        User user = new User(tweetLocation); //original user of the tweet
                        retweetedStatus = new RetweetedStatus(tweetId, tweetString, user);
                    } //if the tweet is a retweet, it will contain this bundle, if not this bundle will be empty
                } else { //if this tweet has not been retweeted
                    tweetId = tweetJsonObject.get("id_str").getAsString(); //twitter recommmends using the string version of the id
                    tweetString = tweetJsonObject.get("full_text").getAsString().trim(); //get the tweet without leading and trailing white spaces
                    JsonObject userJsonObject = tweetJsonObject.get("user").getAsJsonObject(); //we are getting the user because the location is inside this json object
                    tweetLocation = userJsonObject.get("location").getAsString(); //get the tweet location
                }
                int inFavorOfMigration = -2; //-2 means not searched yet. This is for both cases
                Tweet tweet = new Tweet(tweetId, tweetString, tweetLocation, inFavorOfMigration, retweetedStatus);
                allTweets.add(tweet);
            }
        }
        con.close();
        return allTweets;
    }

    public static void addTweetsFromDatabaseToDataset() throws SQLException, IOException, InvalidFormatException {
        Connection con = JDBCConnectionManager.getConnection();
        ArrayList<Tweet> allTweets = getAllTweets();
        ArrayList<String> alreadyAddedTweetsIds = new ArrayList<String>();
        Workbook workbook = WorkbookFactory.create(new File(ArabicDataset.class.getClassLoader().getResource("ar_dataset.xlsx").getFile()));
        Sheet sheet = workbook.getSheetAt(0);
        int currentRowIndex = 1;

        for (int i = 0; i < allTweets.size(); i++) {
            RetweetedStatus retweetedStatus = allTweets.get(i).getRetweetedStatus();

            boolean isEmpty = retweetedStatus.isEmpty(); //to determine: not a retweet
            if (isEmpty) {
                System.out.println("This is an original tweet whose ID: " + allTweets.get(i).getTweetId());
                System.out.println("The tweet: " + allTweets.get(i).getTweetString());
            } else {
                System.out.println("This is a retweet of a tweet whose original ID: " + retweetedStatus.getOriginalTweetId());
                System.out.println("Original Tweet: " + retweetedStatus.getOriginalTweetString());
            }

            //TODO this is getting the dataset from target/classes/ar_dataset.xlsx, change it to take it from resources folder
            Row row = sheet.createRow(i + currentRowIndex); //creates a new row

            Cell tweetIdCell = row.createCell(0); //creates a new cell
            tweetIdCell.setCellType(CellType.STRING); //set the type of the cell to string so that we avoid errors
            Cell tweetStringCell = row.createCell(1); //creates a new cell
            tweetStringCell.setCellType(CellType.STRING); //set the type of the cell to string so that we avoid errors
            Cell tweetLocationCell = row.createCell(2); //creates a new cell
            tweetLocationCell.setCellType(CellType.STRING); //set the type of the cell to string so that we avoid errors

            if (!isEmpty && !alreadyAddedTweetsIds.contains(retweetedStatus.getOriginalTweetId())) { //now we want to check if it is a retweet, put the original tweet in the dataset
                tweetIdCell.setCellValue(retweetedStatus.getOriginalTweetId()); //added the original tweet id to the dataset
                alreadyAddedTweetsIds.add(retweetedStatus.getOriginalTweetId()); //added it to the list to not re-add it again later
                tweetStringCell.setCellValue(retweetedStatus.getOriginalTweetString()); //added the original tweet string to the dataset
                tweetLocationCell.setCellValue(retweetedStatus.getOriginalUser().getLocation()); //added the original tweet location to the dataset
            } else if (isEmpty && !alreadyAddedTweetsIds.contains(allTweets.get(i).getTweetId())) { //if this tweet has not been retweeted
                tweetIdCell.setCellValue(allTweets.get(i).getTweetId()); //added the tweet id to the dataset
                alreadyAddedTweetsIds.add(allTweets.get(i).getTweetId()); //added it to the list to not re-add it again later
                tweetStringCell.setCellValue(allTweets.get(i).getTweetString()); //added the tweet string to the dataset
                tweetLocationCell.setCellValue(allTweets.get(i).getTweetLocation()); //added the tweet location to the dataset
            }
                /*Cell inFavorOfMigrationCell = row.getCell(3); //cell of the in favor of migration
                inFavorOfMigrationCell.setCellType(CellType.NUMERIC); //set the type of the cell to numeric so that we avoid errors
                inFavorOfMigrationCell.setCellValue(allTweets.get(i).getInFavorOfMigration()); //added the in favor of migration to the dataset*/

            else currentRowIndex--; //if an index was skipped do not add an empty line
            System.out.println(alreadyAddedTweetsIds.toString());
            System.out.println();
        }
        System.out.println("Original Tweets size: "+allTweets.size());
        System.out.println("Dataset Tweets size: "+alreadyAddedTweetsIds.size());
        int discardedTweets=allTweets.size()-alreadyAddedTweetsIds.size();
        System.out.println("Discarded Tweets size: "+discardedTweets+" => "+(((double)discardedTweets/(double)allTweets.size())*100)+"% loss");

        FileOutputStream fileOut = new FileOutputStream("ArabicJsonFiles/newar_dataset.xlsx");
        workbook.write(fileOut);
        fileOut.close();
        workbook.close();

        con.close();
    }


}
