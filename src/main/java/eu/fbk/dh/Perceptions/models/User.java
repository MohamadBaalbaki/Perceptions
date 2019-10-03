package eu.fbk.dh.Perceptions.models;

/**
 * @author Mohamad Baalbaki
 */


public class User {
    private String originalTweetLocation;

    public User(){}
    
    public User(String originalTweetLocation){
        this.originalTweetLocation=originalTweetLocation;
    }

    public String getLocation() {
        return originalTweetLocation;
    }

    public void setLocation(String originalTweetLocation) {
        this.originalTweetLocation = originalTweetLocation;
    }

    @Override
    public String toString() {
        return "User{" +
                "originalTweetLocation='" + originalTweetLocation + '\'' +
                '}';
    }
}
