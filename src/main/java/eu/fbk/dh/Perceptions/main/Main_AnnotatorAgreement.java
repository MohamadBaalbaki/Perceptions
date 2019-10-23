package eu.fbk.dh.Perceptions.main;

import eu.fbk.dh.Perceptions.datasets.AnnotatorAgreement;
import org.apache.poi.openxml4j.exceptions.InvalidFormatException;

import java.io.IOException;

/**
 * @author Mohamad Baalbaki
 */


public class Main_AnnotatorAgreement {
    public static void main(String[] args) throws IOException, InvalidFormatException {
        AnnotatorAgreement.getAnnotatorAgreements();
    }
}
