package eu.fbk.dh.Perceptions.database;

import java.io.InputStream;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.Properties;


/**
 * @author Mohamad Baalbaki
 */


public class JDBCConnectionManager {
    private static String username=null;
    private static String password=null;
    private static String connection=null;

    public static Connection getConnection() {
        Connection con=null;
        try {
            Class.forName("com.mysql.cj.jdbc.Driver");
            try {
                try {
                    InputStream input = JDBCConnectionManager.class.getClassLoader().getResourceAsStream("config.properties");
                    Properties prop = new Properties();
                    prop.load(input);
                    username = prop.getProperty("mysqlUser");
                    password= prop.getProperty("mysqlPassword");
                    connection= prop.getProperty("connection");
                }
                catch (Exception e){
                    e.printStackTrace();
                }
                con = DriverManager.getConnection(connection, username, password);
            } catch (SQLException ex) {
                System.out.println("Failed to create the database connection.");
            }
        } catch (ClassNotFoundException ex) {
            System.out.println("Driver not found.");
        }
        return con;
    }
}
