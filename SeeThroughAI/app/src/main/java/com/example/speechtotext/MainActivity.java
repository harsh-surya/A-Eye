package com.example.speechtotext;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.os.Vibrator;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.speech.tts.TextToSpeech;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.Toast;

import com.airbnb.lottie.LottieAnimationView;

import java.util.ArrayList;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {

    ImageButton imageButton;
    //EditText editText;
    LottieAnimationView LottieButton;
    int count = 0;
    static int key = -1;
    static Vibrator vibrator;
    SpeechRecognizer speechRecognizer;
    TextToSpeech textToSpeech;

    private void switchActivities() {

        Intent switchActivityIntent = new Intent(this, Camera.class);
        vibrator.vibrate(1000);
        startActivity(switchActivityIntent);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        textToSpeech = new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if(status == TextToSpeech.SUCCESS)
                {
                    int lang = textToSpeech.setLanguage(Locale.ENGLISH);
                }
            }
        });

        imageButton = findViewById(R.id.button);
        LottieButton=findViewById(R.id.lotteloadID);
        //editText = findViewById(R.id.edittext);
        vibrator = (Vibrator)getSystemService(VIBRATOR_SERVICE);
        if(ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED)
        {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECORD_AUDIO}, 1);
        }
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this);
        final Intent speechRecognizerIntent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        int speech = textToSpeech.speak("Tap to speak", TextToSpeech.QUEUE_FLUSH, null);
        imageButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                SwitchLottie(speechRecognizerIntent);

            }
        });
        LottieButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                SwitchLottie(speechRecognizerIntent);
            }
        });


        speechRecognizer.setRecognitionListener(new RecognitionListener() {
            @Override
            public void onReadyForSpeech(Bundle params) {

            }

            @Override
            public void onBeginningOfSpeech() {

            }

            @Override
            public void onRmsChanged(float rmsdB) {

            }

            @Override
            public void onBufferReceived(byte[] buffer) {

            }

            @Override
            public void onEndOfSpeech() {

            }

            @Override
            public void onError(int error) {

            }

            @Override
            public void onResults(Bundle results) {
                ArrayList<String> data = results.getStringArrayList(speechRecognizer.RESULTS_RECOGNITION);
                //editText.setText(data.get(0));
                if(data.get(0).contains("currency"))
                {
                    key =0;
                }else if(data.get(0).contains("speak to me"))
                {
                    key =1;
                }else if(data.get(0).contains("document"))
                {
                    key =2;
                }
                switchActivities();
            }

            @Override
            public void onPartialResults(Bundle partialResults) {

            }

            @Override
            public void onEvent(int eventType, Bundle params) {

            }
        });
    }

    private void SwitchLottie(Intent speechRecognizerIntent) {
        if(count == 0)
        {

            LottieButton.setVisibility(View.VISIBLE);

            //start listening
            MediaPlayer mp = MediaPlayer.create(MainActivity.this, R.raw.speaknow);
            mp.start();
            vibrator.vibrate(1000);
            speechRecognizer.startListening(speechRecognizerIntent);
            count = 1;


        }else
        {

            LottieButton.setVisibility(View.INVISIBLE);
            //stop listening
            speechRecognizer.stopListening();
            vibrator.vibrate(1000);
            count = 0;
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if(requestCode == 1)
        {
            if(grantResults[0] == PackageManager.PERMISSION_GRANTED)
            {
                Toast.makeText(this, "Permission Granted", Toast.LENGTH_SHORT).show();
            }else
            {
                Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show();
            }
        }
    }
}
