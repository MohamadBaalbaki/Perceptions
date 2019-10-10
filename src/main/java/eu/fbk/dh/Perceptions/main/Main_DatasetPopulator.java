package eu.fbk.dh.Perceptions.main;

import eu.fbk.dh.Perceptions.database.JDBCConnectionManager;
import eu.fbk.dh.Perceptions.datasets.ArabicDataset;
import org.apache.poi.openxml4j.exceptions.InvalidFormatException;

import java.io.IOException;
import java.sql.Connection;
import java.sql.SQLException;

/**
 * @author Mohamad Baalbaki
 */


public class Main_DatasetPopulator {
    public static void main(String[] args) throws InvalidFormatException, SQLException, IOException {
        Connection con = JDBCConnectionManager.getConnection();

        //Todo: add tweets to a dataset in an excel file (discard repetitions) and manually classify them
        ArabicDataset.addTweetsFromDatabaseToDataset(); //load the tweets and their information to the dataset from the database

        con.close();
    }
}
