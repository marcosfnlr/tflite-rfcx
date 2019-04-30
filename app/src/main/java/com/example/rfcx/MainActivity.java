package com.example.rfcx;

import android.content.res.AssetFileDescriptor;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    public void start(View view) {
        MappedByteBuffer model = null;
        try {
            model = loadModelFile("conv_actions_frozen.tflite");
        }catch (IOException e){
            e.printStackTrace();
        }
        try(Interpreter tflite = new Interpreter(model, new Interpreter.Options())){
            //tflite.runForMultipleInputsOutputs();
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
