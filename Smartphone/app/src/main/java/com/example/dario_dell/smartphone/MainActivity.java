package com.example.dario_dell.smartphone;

import android.os.Bundle;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.MotionEvent;
import android.view.VelocityTracker;
import android.widget.TextView;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.time.LocalDate;

public class MainActivity extends AppCompatActivity implements ViewWasTouchedListener {

    private TextView velocity_x_txtView, velocity_y_txtView,
            maxVelocity_x_txtView, maxVelocity_y_txtView, velocity_magnitude_txtView;
    private DrawingView drawing;

    //needed for providing unique names to the files
    private int filename_counter = 0;

    //along with date and counter, they provide unique names to the files
    private final String filename_separator = "_";
    private final String filename_suffix = "smartphone_sample";
    private final String filename_format = ".csv";
    private String folder_name = "/Drawing_Data/";

    // resulting string to write into a CSV file
    // Init: CSV file header
    String to_write = "timestamp,x,y,x_velocity,y_velocity\n";
    String text_separator = ",";

    private VelocityTracker velocityTracker = null;
    private float max_velocity_x, max_velocity_y;

    // needed for inch to meter conversion
    DisplayMetrics metrics = null;
    private final float inchToMeterRatio = (float) 39.3701;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        velocity_x_txtView = (TextView) findViewById(R.id.velocityx);
        velocity_y_txtView = (TextView) findViewById(R.id.velocityy);
        maxVelocity_x_txtView = (TextView) findViewById(R.id.maxvelocityx);
        maxVelocity_y_txtView = (TextView) findViewById(R.id.maxvelocityy);
        velocity_magnitude_txtView = (TextView) findViewById(R.id.velocity_magnitude);
        drawing = (DrawingView) findViewById(R.id.drawing);
        drawing.setWasTouchedListener(this);
        resetView();
    }


    @Override
    //callback for updating the TextViews
    public void onViewTouched(float x, float y, MotionEvent event) {

        int action = event.getActionMasked();
        metrics = getResources().getDisplayMetrics();
        System.out.println(metrics.xdpi*metrics.ydpi);
        String to_add = "";

        switch (action){
            //A pressed gesture has started, the motion contains the initial starting location.
            case MotionEvent.ACTION_DOWN:
                if (velocityTracker == null)
                    velocityTracker = VelocityTracker.obtain();
                else
                    velocityTracker.clear();

                //Add a user's movement to the tracker.
                // This call should be done for the initial ACTION_DOWN,
                // the following ACTION_MOVE events that you receive,
                // and the final ACTION_UP.
                velocityTracker.addMovement(event);
                max_velocity_x = 0;
                max_velocity_y = 0;
                resetView();
                break;

            case MotionEvent.ACTION_MOVE:
                velocityTracker.addMovement(event);
                //pixels/msec - - - change the parameter if you need different units
                velocityTracker.computeCurrentVelocity(10);

                float velocity_x = velocityTracker.getXVelocity();
                float velocity_y = velocityTracker.getYVelocity();

                if (velocity_x > max_velocity_x)
                    max_velocity_x = velocity_x;

                if (velocity_y > max_velocity_y)
                    max_velocity_y = velocity_y;

                double magnitude = calculateMagnitude(velocity_x,velocity_y);

                updateView(velocity_x, velocity_y, max_velocity_x, max_velocity_y, magnitude);
                to_write += System.currentTimeMillis() + text_separator +
                        x/metrics.xdpi/inchToMeterRatio + text_separator +
                        y/metrics.ydpi/inchToMeterRatio + text_separator +
                        Float.toString(toMeterPerSecondsConversion(velocity_x, metrics.xdpi)) + text_separator +
                        Float.toString(toMeterPerSecondsConversion(velocity_y, metrics.ydpi)) + "\n";
                break;

            case MotionEvent.ACTION_UP:
            case MotionEvent.ACTION_CANCEL:
                velocityTracker.addMovement(event);
                velocityTracker.recycle();
                velocityTracker = null;
                writeToFile();
                to_write = "timestamp,x,y,x_velocity,y_velocity\n";
                resetView();
                break;
        }
        // Calling invalidate will cause the onDraw method to execute
        this.findViewById(android.R.id.content).invalidate();
    }


    protected static double calculateMagnitude(float x, float y){
        return Math.sqrt(Math.pow(x, 2) + Math.pow(y, 2));
    }


    private void updateView(float velocity_x, float velocity_y,
                            float max_velocity_x, float max_velocity_y, double velocity_magnitude){
        velocity_x_txtView.setText("X-velocity: " + Float.toString(velocity_x));
        velocity_y_txtView.setText("Y-velocity: " + Float.toString(velocity_y));
        maxVelocity_x_txtView.setText("max. X-velocity: " + Float.toString(max_velocity_x));
        maxVelocity_y_txtView.setText("max. Y-velocity: " + Float.toString(max_velocity_y));
        velocity_magnitude_txtView.setText("Velocity magnitude: " + Double.toString(velocity_magnitude));
    }


    private void resetView(){
        updateView(0,0,0,0, 0);
    }


    private void resetXYVelocityAndMagnitudeViews(float max_velocity_x, float max_velocity_y,
                                                  double velocity_magnitude){
        updateView(0,0, max_velocity_x, max_velocity_y, 0);
    }


    //generates sequenced unique filenames
    private String generateSequencedFilename(File path){
        String result = LocalDate.now().toString() + filename_separator
                + generateFilenameCounter(path) + filename_separator
                + filename_suffix + filename_format;
        return result;
    }


    /* Checks if external storage is available for read and write */
    /* Taken from the Android official documentation */
    private boolean isExternalStorageWritable() {
        String state = Environment.getExternalStorageState();
        if (Environment.MEDIA_MOUNTED.equals(state)) {
            return true;
        }
        return false;
    }


    //Method for writing to file after removing the finger form the screen
    private void writeToFile() {
        if (MainActivity.this.isExternalStorageWritable()){
            File path = new File (Environment.getExternalStoragePublicDirectory(
                    Environment.MEDIA_SHARED), folder_name);

            FileOutputStream stream = null;
            try {
                boolean r = path.mkdirs();
                if (!r && !path.exists())
                    throw new FileNotFoundException("DIRECTORY NOT CREATED!");

                File file = new File(path, generateSequencedFilename(path));
                if (!file.createNewFile() && !file.exists())
                    throw new FileNotFoundException("FILE NOT CREATED!");
                Log.d("MFILE",file.exists()+ " "+ file.toString());

                stream = new FileOutputStream(file);
                if (to_write.length()==0)
                    throw new FileNotFoundException("STRING TOO SMALL!");
                stream.write(to_write.getBytes());
                stream.flush();
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                try {
                    stream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private int generateFilenameCounter(File path) {
        File folder = new File(path.toString());
        File[] listOfFiles = folder.listFiles();
        if (listOfFiles.length > 0) {
            for (int i = 0; i < listOfFiles.length; i++) {
                if (listOfFiles[i].isFile()) {
                    String filename = listOfFiles[i].getName();
                    String[] filename_parts = filename.split(filename_separator);

                    // String example: 2018-06-07_2_smartphone_sample
                    int sequence_number = Integer.parseInt(
                            filename_parts[filename_parts.length - 3]);

                    filename_counter = Math.max(sequence_number, filename_counter);
                }
            }
        }
        return ++filename_counter;
    }

    // velocityTracker returns pixels/msecs => need to convert to m/sec
    private float toMeterPerSecondsConversion (float velocity, float dpi){
        //return velocity/dpi/inchToMeterRatio * 1000;   old wrong verion
        return velocity*1000*inchToMeterRatio/dpi;
    }

}