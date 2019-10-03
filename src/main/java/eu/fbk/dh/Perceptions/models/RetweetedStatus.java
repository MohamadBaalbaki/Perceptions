package eu.fbk.dh.Perceptions.models;

/**
 * @author Mohamad Baalbaki
 */


public class RetweetedStatus {
    private String originalTweetId;
    private String originalTweetString;
    private User originalUser;
    private boolean empty;

    public RetweetedStatus(){
        this.empty=true;
    }

    public RetweetedStatus(String originalTweetId, String originalTweetString,User originalUser){
        this.originalTweetId = originalTweetId;
        this.originalTweetString=originalTweetString;
        this.originalUser=originalUser;
        this.empty=false;
    }

    public String getOriginalTweetId() {
        return originalTweetId;
    }

    public void setOriginalTweetId(String originalTweetId) {
        this.originalTweetId = originalTweetId;
    }

    public String getOriginalTweetString() {
        return originalTweetString;
    }

    public void setOriginalTweetString(String originalTweetString) {
        this.originalTweetString = originalTweetString;
    }

    public User getOriginalUser() {
        return originalUser;
    }

    public void setOriginalUser(User originalUser) {
        this.originalUser = originalUser;
    }

    public boolean isEmpty() {
        return empty;
    }

    @Override
    public String toString() {
        return "RetweetedStatus{" +
                "originalTweetId='" + originalTweetId + '\'' +
                ", originalTweetString='" + originalTweetString + '\'' +
                ", originalUser=" + originalUser +
                '}';
    }
}
