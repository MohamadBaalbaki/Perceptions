package eu.fbk.dh.Perceptions.models;

/**
 * @author Mohamad Baalbaki
 */


public class Tweet {
    private String tweetId;
    private String tweetString;
    private String tweetLocation;
    private int inFavorOfMigration;
    private RetweetedStatus retweetedStatus;

    public Tweet(String tweetId, String tweetString, String tweetLocation, int inFavorOfMigration, RetweetedStatus retweetedStatus) {
        this.tweetId = tweetId;
        this.tweetString = tweetString;
        this.tweetLocation = tweetLocation;
        this.inFavorOfMigration = inFavorOfMigration;
        this.retweetedStatus = retweetedStatus;
    }

    public String getTweetId() {
        return tweetId;
    }

    public void setTweetId(String tweetId) {
        this.tweetId = tweetId;
    }

    public String getTweetString() {
        return tweetString;
    }

    public void setTweetString(String tweetString) {
        this.tweetString = tweetString;
    }

    public String getTweetLocation() {
        return tweetLocation;
    }

    public void setTweetLocation(String tweetLocation) {
        this.tweetLocation = tweetLocation;
    }

    public int getInFavorOfMigration() {
        return inFavorOfMigration;
    }

    public void setInFavorOfMigration(int inFavorOfMigration) {
        this.inFavorOfMigration = inFavorOfMigration;
    }

    public RetweetedStatus getRetweetedStatus() {
        return retweetedStatus;
    }

    public void setRetweetedStatus(RetweetedStatus retweetedStatus) {
        this.retweetedStatus = retweetedStatus;
    }

    @Override
    public String toString() {
        return "Tweet{" +
                "tweetId='" + tweetId + '\'' +
                ", tweetString='" + tweetString + '\'' +
                ", tweetLocation='" + tweetLocation + '\'' +
                ", inFavorOfMigration=" + inFavorOfMigration +
                ", retweetedStatus=" + retweetedStatus +
                '}';
    }
}
