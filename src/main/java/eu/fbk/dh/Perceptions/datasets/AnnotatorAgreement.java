package eu.fbk.dh.Perceptions.datasets;

import org.apache.poi.openxml4j.exceptions.InvalidFormatException;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.ss.usermodel.WorkbookFactory;

import java.io.File;
import java.io.IOException;

/**
 * @author Mohamad Baalbaki
 */


public class AnnotatorAgreement {

    //Todo change getting files from class to resources
    public static void getAnnotatorAgreements() throws IOException, InvalidFormatException {
        Workbook baalbakiWorkbook = WorkbookFactory.create(new File(ArabicDataset.class.getClassLoader().getResource("baalbaki_annotation.xlsx").getFile()));
        Sheet baalbakiSheet = baalbakiWorkbook.getSheetAt(0);
        Workbook harmoushWorkbook = WorkbookFactory.create(new File(ArabicDataset.class.getClassLoader().getResource("harmoush_annotation.xlsx").getFile()));
        Sheet harmoushSheet = harmoushWorkbook.getSheetAt(0);
        int counter1and1=0;
        int counter1and0=0;
        int counter1andMinus1=0;
        int counter0and1=0;
        int counter0and0=0;
        int counter0andMinus1=0;
        int counterMinus1and1=0;
        int counterMinus1and0=0;
        int counterMinus1andMinus1=0;
        int counterDisagreements=0;

        for(int i=1;i<baalbakiSheet.getPhysicalNumberOfRows();i++){
            //System.out.println((int)baalbakiSheet.getRow(i).getCell(3).getNumericCellValue());
            int baalbakiAnnotation=(int)baalbakiSheet.getRow(i).getCell(3).getNumericCellValue();
            int harmoushAnnotation=(int)harmoushSheet.getRow(i).getCell(3).getNumericCellValue();

            ///////////////////ONLY TO COUNT DISAGREEMENTS AND TO SHOW THEIR ROWS////////////////////////////////
            /*if(baalbakiAnnotation!=harmoushAnnotation){
                System.out.println("Sheet row: "+(i+1)+"=> Baalbaki: "+baalbakiAnnotation+" | Harmoush: "+harmoushAnnotation);
                counterDisagreements++;
            }*/
            ////////////////////UNCOMMENT WITH THE LOWER COMMENTED SOUT SECTION//////////////////////////////////

            if(baalbakiAnnotation==1 && harmoushAnnotation==1){ // 1 and 1
                counter1and1++;
                continue;
            }
            else if (baalbakiAnnotation == 1 && harmoushAnnotation == 0){ // 1 and 0
                counter1and0++;
                counterDisagreements++;
                continue;
            }
            else if (baalbakiAnnotation == 1 && harmoushAnnotation == -1){ // 1 and -1
                counter1andMinus1++;
                counterDisagreements++;
                continue;
            }
            else if (baalbakiAnnotation==0 && harmoushAnnotation==1){ // 0 and 1
                counter0and1++;
                counterDisagreements++;
                continue;
            }
            else if (baalbakiAnnotation==0 && harmoushAnnotation==0){ // 0 and 0
                counter0and0++;
                continue;
            }
            else if (baalbakiAnnotation==0 && harmoushAnnotation==-1){ // 0 and -1
                counter0andMinus1++;
                counterDisagreements++;
                continue;
            }
            else if (baalbakiAnnotation==-1 && harmoushAnnotation==1){ // -1 and 1
                counterMinus1and1++;
                counterDisagreements++;
                continue;
            }
            else if (baalbakiAnnotation==-1 && harmoushAnnotation==0){ // -1 and 0
                counterMinus1and0++;
                counterDisagreements++;
                continue;
            }
            else if (baalbakiAnnotation==-1 && harmoushAnnotation==-1){ // -1 and -1
                counterMinus1andMinus1++;
                continue;
            }
        }

        System.out.println();
        System.out.println("1 and 1: "+counter1and1+"\n"+"1 and 0: "+counter1and0+"\n"+"1 and -1: "+counter1andMinus1+
                "\n"+"0 and 1: "+counter0and1+"\n"+"0 and 0: "+counter0and0+"\n"+"0 and -1: "+counter0andMinus1+
                "\n"+"-1 and 1: "+counterMinus1and1+"\n"+"-1 and 0: "+counterMinus1and0+"\n"+"-1 and -1: "+counterMinus1andMinus1+"\n"+
                "\n"+"TOTAL DISAGREEMENTS: "+counterDisagreements);

        ///////////////////////UNCOMMENT WITH THE UPPER SECTION AND COMMENT ALL THE REST////////////////////////
        //System.out.println("Total disagreements: "+counterDisagreements);
    }
}
