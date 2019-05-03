package com.example.rfcx;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class MainActivity extends AppCompatActivity {

    private final String LABELS_FILE = "conv_actions_labels.txt";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    public void start(View view) {
        MappedByteBuffer model = null;
        List<String> labels = new ArrayList<String>();
        try {
            model = loadModelFile("conv_actions_frozen.tflite");
        } catch (IOException e) {
            throw new RuntimeException("Problem reading model file!", e);
        }
        try (BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open(LABELS_FILE)));) {
            String line;
            while ((line = br.readLine()) != null) {
                labels.add(line);
            }
        } catch (IOException e) {
            throw new RuntimeException("Problem reading label file!", e);
        }
        try (Interpreter tflite = new Interpreter(model, new Interpreter.Options())) {
            float[][] floatInputBuffer = new float[16000][1];
            int[] sampleRateList = new int[]{16000};

            Random random = new Random();
            for (int i = 0; i < 16000; ++i) {
                floatInputBuffer[i][0] = (random.nextInt(Short.MAX_VALUE)) / 32767.0f;
            }

            float[][] outputScores = new float[1][labels.size()];
            Map<Integer, Object> outputMap = new HashMap<>();
            outputMap.put(0, outputScores);
            Object[] inputArray = {floatInputBuffer, sampleRateList};

            tflite.runForMultipleInputsOutputs(inputArray, outputMap);

            TextView textView = findViewById(R.id.textView);
            textView.setText(Float.toString(outputScores[0][0]));
        } catch (Exception e) {
            throw new RuntimeException("Problem executing model!", e);
        }

    }

    private MappedByteBuffer loadModelFile(String modelPath) throws IOException {
        try (AssetFileDescriptor fileDescriptor = getApplicationContext().getAssets().openFd(modelPath);
             FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }
}
