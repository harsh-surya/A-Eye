package com.example.speechtotext;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import java.util.ArrayList;
import java.util.List;
import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Point;
import android.graphics.Rect;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.TextRecognizer;
import com.google.mlkit.vision.text.TextRecognizerOptions;

import com.google.mlkit.vision.text.Text;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;


import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;


public class Result extends AppCompatActivity {

    ImageView selectedImage;
    String path = "";
    List<String> imagesList;
    TextToSpeech textToSpeech;
    TextView textView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);

        textView = (TextView) findViewById(R.id.textView);
        textView.setMovementMethod(new ScrollingMovementMethod());

        selectedImage = findViewById(R.id.displayImageView);
        path = Camera.currentPhotoPath.substring(1);
        imagesList = new ArrayList<String>();
        imagesList.add(path);

        selectedImage.setImageURI(Uri.fromFile(new File(path)));
        uploadFiles();
    }

    public void uploadFiles() {
        if(imagesList.size() == 0) {
            Toast.makeText(this, "Can't choose pictures", Toast.LENGTH_SHORT).show();
            return;
        }
        Map<String, RequestBody> files = new HashMap<>();
        textToSpeech = new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if(status == TextToSpeech.SUCCESS)
                {
                    int lang = textToSpeech.setLanguage(Locale.ENGLISH);
                }
            }
        });
        if(MainActivity.key == 0)
        {
            final UploadApis service = NetworkClient.createService(UploadApis.class);
            for (int i = 0; i < imagesList.size(); i++) {
                File file = new File(imagesList.get(i));
                files.put("file" + i + "\"; filename=\"" + file.getName(), RequestBody.create(MediaType.parse("image/jpg"), file));
            }
            Call<UploadResult> call = service.uploadMultipleFiles(files);
            call.enqueue(new Callback<UploadResult>() {
                @Override
                public void onResponse(Call<UploadResult> call, Response<UploadResult> response) {
                    if (response.isSuccessful() && response.body().code == 1) {
                        //Toast.makeText(Result.this, "Upload success", Toast.LENGTH_SHORT).show();

                        textView.setText(response.body().prediction + " Rupees");

                        MainActivity.vibrator.vibrate(1000);
                        int speech = textToSpeech.speak(response.body().prediction + " Rupees", TextToSpeech.QUEUE_FLUSH, null);
                        Log.i("orzangleli", "---------------------Upload success -----------------------");
                        Log.i("orzangleli", "The base address is:" + NetworkClient.API_BASE_URL);
                        Log.i("orzangleli", "The relative address of the picture is:" + response.body().prediction);
                        Log.i("orzangleli", "---------------------END-----------------------");
                    }
                }

                @Override
                public void onFailure(Call<UploadResult> call, Throwable t) {
                    Toast.makeText(Result.this, "upload failed "+t, Toast.LENGTH_SHORT).show();
                }
            });
        }else if(MainActivity.key == 1)
        {
            final CaptionApi service = NetworkClient.createService(CaptionApi.class);
            for (int i = 0; i < imagesList.size(); i++) {
                File file = new File(imagesList.get(i));
                files.put("file" + i + "\"; filename=\"" + file.getName(), RequestBody.create(MediaType.parse("image/jpg"), file));
            }
            Call<UploadResult> call = service.uploadMultipleFiles(files);
            call.enqueue(new Callback<UploadResult>() {
                @Override
                public void onResponse(Call<UploadResult> call, Response<UploadResult> response) {
                    if (response.isSuccessful() && response.body().code == 1) {
                        //Toast.makeText(Result.this, "Upload success", Toast.LENGTH_SHORT).show();

                        textView.setText(response.body().prediction);

                        MainActivity.vibrator.vibrate(1000);
                        int speech = textToSpeech.speak(response.body().prediction, TextToSpeech.QUEUE_FLUSH, null);

                        Log.i("orzangleli", "---------------------Upload success -----------------------");
                        Log.i("orzangleli", "The base address is:" + NetworkClient.API_BASE_URL);
                        Log.i("orzangleli", "The relative address of the picture is:" + response.body().prediction);
                        Log.i("orzangleli", "---------------------END-----------------------");
                    }
                }

                @Override
                public void onFailure(Call<UploadResult> call, Throwable t) {
                    Toast.makeText(Result.this, "upload failed "+t, Toast.LENGTH_SHORT).show();
                }
            });
        }else
        {
            File file = new File(path);

            InputImage image;
            try {
                image = InputImage.fromFilePath(this, Uri.fromFile(file));
                recognizeText(image);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private void recognizeText(InputImage image) {
        TextRecognizer recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);

        Task<Text> result =
                recognizer.process(image)
                        .addOnSuccessListener(new OnSuccessListener<Text>() {
                            @Override
                            public void onSuccess(Text visionText) {
                                String text = "";
                                for (Text.TextBlock block : visionText.getTextBlocks()) {
                                    //Rect boundingBox = block.getBoundingBox();
                                    //Point[] cornerPoints = block.getCornerPoints();
                                    text += block.getText() + "\n";
                                }
                                int speech = textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null);
                                textView.setText(text);
                                MainActivity.vibrator.vibrate(1000);

                                Log.i("orzangleli", "---------------------Upload success -----------------------");
                                Log.i("orzangleli", "The base address is:" + NetworkClient.API_BASE_URL);
                                Log.i("orzangleli", "The relative address of the picture is:" + text);
                                Log.i("orzangleli", "---------------------END-----------------------");
                            }
                        })
                        .addOnFailureListener(
                                new OnFailureListener() {
                                    @Override
                                    public void onFailure(@NonNull Exception e) {
                                        Toast.makeText(Result.this, "Error: "+e.getMessage(), Toast.LENGTH_SHORT).show();
                                    }
                                });
    }
}