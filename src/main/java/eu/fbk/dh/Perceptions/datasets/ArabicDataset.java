package eu.fbk.dh.Perceptions.datasets;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;
import eu.fbk.dh.Perceptions.database.JDBCConnectionManager;
import eu.fbk.dh.Perceptions.models.ExcelFile;

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

    public void addUnseenTweetsInformationFromDatabase() throws SQLException {
        Connection con = JDBCConnectionManager.getConnection();
        PreparedStatement preparedStatement=con.prepareStatement("SELECT allTweets from ar_tweets"); //get all the tweets from the database
        ResultSet resultSet=preparedStatement.executeQuery();
        JsonParser jsonParser=new JsonParser(); //to parse string into json array
        while(resultSet.next()){
            JsonElement jsonElement = jsonParser.parse(resultSet.getString("allTweets"));
            JsonArray tweetsAsJsonArray = jsonElement.getAsJsonArray();
            System.out.println(tweetsAsJsonArray);

        }

        con.close();
    }

}
