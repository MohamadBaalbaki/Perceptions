package eu.fbk.dh.Perceptions.datasets;

import com.google.gson.*;
import eu.fbk.dh.Perceptions.database.JDBCConnectionManager;
import eu.fbk.dh.Perceptions.models.ExcelFile;
import eu.fbk.dh.Perceptions.models.RetweetedStatus;
import eu.fbk.dh.Perceptions.models.Tweet;
import eu.fbk.dh.Perceptions.models.User;
import net.ricecode.similarity.LevenshteinDistanceStrategy;
import net.ricecode.similarity.SimilarityStrategy;
import net.ricecode.similarity.StringSimilarityService;
import net.ricecode.similarity.StringSimilarityServiceImpl;
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
import java.util.Arrays;

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
        ArrayList<String> alreadyAddedTweetsIdsAsList=new ArrayList<>();

        Workbook workbook = WorkbookFactory.create(new File(ArabicDataset.class.getClassLoader().getResource("ar_dataset.xlsx").getFile()));
        Sheet sheet = workbook.getSheetAt(0);

        PreparedStatement getCountOfAlreadyAddedTweetsPS=con.prepareStatement("SELECT COUNT(tweetId) from ar_already_seen_tweets");
        ResultSet getCountOfAlreadyAddedTweetsRS=getCountOfAlreadyAddedTweetsPS.executeQuery();
        int countOfAlreadyAddedTweets=-2; //initializing
        while(getCountOfAlreadyAddedTweetsRS.next()){
            countOfAlreadyAddedTweets=getCountOfAlreadyAddedTweetsRS.getInt(1);
        }
        int startFromRow=countOfAlreadyAddedTweets;

        PreparedStatement getAlreadyAddedTweetsIdsPS=con.prepareStatement("SELECT tweetId from ar_already_seen_tweets");
        ResultSet alreadyAddedTweetsIdsRS=getAlreadyAddedTweetsIdsPS.executeQuery();
        while(alreadyAddedTweetsIdsRS.next()){
            alreadyAddedTweetsIdsAsList.add(alreadyAddedTweetsIdsRS.getString("tweetId")); //local copy of db info
        }

        /*final ArrayList<String> egyptianWords = new ArrayList<String>(Arrays.asList("مصر", "سكندرية", "قاهرة", "جيزة", "egypt", "cairo", "alexandria", "giza", "égypte", "caire", "alexandrie")); //these are the different ways egypt can be written in the location
        final ArrayList<String> algerianWords = new ArrayList<String>(Arrays.asList("جزائر", "وهران", "algiers", "alger", "oran", "algérie", "algeri")); //these are the different ways algeria can be written in the location
        final ArrayList<String> tunisianWords = new ArrayList<String>(Arrays.asList("تونس", "tunis", "sfax")); //these are the different ways tunisia can be written in the location
*/
        for (int i = countOfAlreadyAddedTweets; i < allTweets.size(); i++) {
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

            if (!isEmpty) { //if this tweet is a retweet
                System.out.println("!is Empty and original tweet not added: "+!alreadyAddedTweetsIdsAsList.contains(retweetedStatus.getOriginalTweetId()));
                if (/*(egyptianWords.parallelStream().anyMatch(retweetedStatus.getOriginalUser().getLocation().toLowerCase()::contains) //if the location is known in one of our arraylists (check if the string contains any of the cities in the arraylist)
                        || algerianWords.parallelStream().anyMatch(retweetedStatus.getOriginalUser().getLocation().toLowerCase()::contains)
                        || tunisianWords.parallelStream().anyMatch(retweetedStatus.getOriginalUser().getLocation().toLowerCase()::contains))
                        &&*/ (!alreadyAddedTweetsIdsAsList.contains(retweetedStatus.getOriginalTweetId()))) { //if the original tweet was not already added to the dataset

                    startFromRow++;

                    Row row = sheet.createRow(startFromRow); //creates a new row

                    System.out.println("Created row: " + (startFromRow));

                    Cell tweetIdCell = row.createCell(0); //creates a new cell
                    tweetIdCell.setCellType(CellType.STRING); //set the type of the cell to string so that we avoid errors
                    Cell tweetStringCell = row.createCell(1); //creates a new cell
                    tweetStringCell.setCellType(CellType.STRING); //set the type of the cell to string so that we avoid errors
                    Cell tweetLocationCell = row.createCell(2); //creates a new cell
                    tweetLocationCell.setCellType(CellType.STRING); //set the type of the cell to string so that we avoid errors

                    tweetIdCell.setCellValue(retweetedStatus.getOriginalTweetId()); //added the original tweet id to the dataset
                    alreadyAddedTweetsIdsAsList.add(retweetedStatus.getOriginalTweetId()); //added it to the list to not re-add it again later
                    PreparedStatement addTweetIdToDbPS=con.prepareStatement("INSERT INTO ar_already_seen_tweets values (?)");
                    addTweetIdToDbPS.setString(1,retweetedStatus.getOriginalTweetId());
                    addTweetIdToDbPS.executeUpdate();

                    tweetStringCell.setCellValue(retweetedStatus.getOriginalTweetString()); //added the original tweet string to the dataset
                    tweetLocationCell.setCellValue(retweetedStatus.getOriginalUser().getLocation()); //added the original tweet location to the dataset

                } else {
                    System.out.println("SKIPPED FOR NOT SATISFYING CONDITIONS");
                }
            } else if (isEmpty) { //if this tweet is not a retweet
                System.out.println("is Empty and original tweet not added: "+!alreadyAddedTweetsIdsAsList.contains(retweetedStatus.getOriginalTweetId()));
                if (/*(egyptianWords.parallelStream().anyMatch(allTweets.get(i).getTweetLocation().toLowerCase()::contains)
                        || algerianWords.parallelStream().anyMatch(allTweets.get(i).getTweetLocation().toLowerCase()::contains)
                        || tunisianWords.parallelStream().anyMatch(allTweets.get(i).getTweetLocation().toLowerCase()::contains))
                        &&*/ (!alreadyAddedTweetsIdsAsList.contains(allTweets.get(i).getTweetId()))) {

                    startFromRow++;

                    Row row = sheet.createRow(startFromRow); //creates a new row

                    System.out.println("Created row: " + (startFromRow));

                    Cell tweetIdCell = row.createCell(0); //creates a new cell
                    tweetIdCell.setCellType(CellType.STRING); //set the type of the cell to string so that we avoid errors
                    Cell tweetStringCell = row.createCell(1); //creates a new cell
                    tweetStringCell.setCellType(CellType.STRING); //set the type of the cell to string so that we avoid errors
                    Cell tweetLocationCell = row.createCell(2); //creates a new cell
                    tweetLocationCell.setCellType(CellType.STRING); //set the type of the cell to string so that we avoid errors

                    tweetIdCell.setCellValue(allTweets.get(i).getTweetId()); //added the tweet id to the dataset
                    alreadyAddedTweetsIdsAsList.add(allTweets.get(i).getTweetId()); //added it to the list to not re-add it again later
                    PreparedStatement addTweetIdToDbPS=con.prepareStatement("INSERT INTO ar_already_seen_tweets values (?)");
                    addTweetIdToDbPS.setString(1,allTweets.get(i).getTweetId());
                    addTweetIdToDbPS.executeUpdate();
                    tweetStringCell.setCellValue(allTweets.get(i).getTweetString()); //added the tweet string to the dataset
                    tweetLocationCell.setCellValue(allTweets.get(i).getTweetLocation()); //added the tweet location to the dataset

                } else {
                    System.out.println("SKIPPED FOR NOT SATISFYING CONDITIONS");
                }
            }

            System.out.println(alreadyAddedTweetsIdsAsList.toString());
            System.out.println();
        }

        System.out.println("Total number of crawled tweets: " + allTweets.size());
        System.out.println("Total number of relevant tweets added to dataset: " + alreadyAddedTweetsIdsAsList.size());
        int discardedTweets = allTweets.size() - alreadyAddedTweetsIdsAsList.size();
        System.out.println("Total number of irrelevant discarded tweets: " + discardedTweets + " => " + (((double) discardedTweets / (double) allTweets.size()) * 100) + "% loss");

        FileOutputStream fileOut = new FileOutputStream("ArabicJsonFiles/newar_dataset.xlsx");
        workbook.write(fileOut);
        fileOut.close();
        workbook.close();

        con.close();
    }


    public static void getTweetsStringSimilarities() throws IOException, InvalidFormatException {
        ExcelFile excelFile = new ExcelFile(new File("/home/baalbaki/Desktop/checkrepetitions.xlsx"));
        XSSFSheet sheet = excelFile.getSheet();
        int numberOfRows = excelFile.getRows();
        int numberOfColumns = sheet.getRow(0).getPhysicalNumberOfCells();
        System.out.println(numberOfRows);

        SimilarityStrategy strategy = new LevenshteinDistanceStrategy();
        StringSimilarityService service = new StringSimilarityServiceImpl(strategy);

        for (int i = 1; i < numberOfRows; i++) {
            for (int j = i+1; j < numberOfRows; j++) {
                //System.out.println(i+" , "+j);
                String firstTweet=sheet.getRow(i).getCell(1).toString();
                String secondTweet=sheet.getRow(j).getCell(1).toString();
                double similarityScore = service.score(firstTweet, secondTweet);
                if(similarityScore>0.5){
                    System.out.println("FOUND!!!");
                    System.out.println(similarityScore);
                    System.out.println("Rows: i="+(i+1)+", j="+(j+1));
                    System.out.println(firstTweet);
                    System.out.println();
                    System.out.println(secondTweet);
                    System.out.println("--------------------------------------");
                    System.out.println();
                }

                /*int pos = (int) Math.round(sheet.getRow(i).getCell(j).getNumericCellValue());
                int neg = (int) Math.round(sheet.getRow(i).getCell(j + 1).getNumericCellValue());
                String word = sheet.getRow(i).getCell(1).toString();

                //System.out.println(word+": "+pos+", "+neg);
                if (neg == 1 && pos == 0) { //pos==0 is not necessary but just to make sure
                    negativeWords.add(word.toLowerCase());
                } else if (pos == 1 && neg == 0) {
                    positiveWords.add(word.toLowerCase());
                }*/
            }
        }

    }
}
