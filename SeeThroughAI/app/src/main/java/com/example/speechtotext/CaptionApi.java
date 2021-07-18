package com.example.speechtotext;

import java.util.Map;

import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.PartMap;

public interface CaptionApi {
    @Multipart
    @POST("/caption")
    Call<UploadResult> uploadMultipleFiles(
            @PartMap Map<String, RequestBody> files);
}
